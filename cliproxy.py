"""cliproxy-py — lightweight OpenAI-compatible proxy (stdlib only).

GET  /healthz, /, /v1/models — POST /v1/chat/completions, /v1/completions,
/v1/embeddings, /v1/responses. Multi-upstream round-robin/fill-first with
failover + cooldown, SSE passthrough, per-upstream keys. OpenCode Go
(opencode.ai) upstreams get the required `x-opencode-session` header.
"""
from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

VERSION = "0.2.0"
DEFAULT_PORT = 8317

# OpenCode Zen endpoints (e.g. https://opencode.ai/zen/go/v1) reject requests
# without a per-conversation routing header (`400 MissingSessionID`). Per
# earendil-works/pi#10820 injection is keyed off the endpoint hostname, not a
# provider id; a caller-supplied header always wins.
OPENCODE_SESSION_HEADER = "x-opencode-session"
OPENCODE_HOST = "opencode.ai"
_AUTO_SESSION_ID = uuid.uuid4().hex


def is_opencode_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        return urlparse(base_url).hostname == OPENCODE_HOST
    except (ValueError, AttributeError, TypeError):
        return False


def resolve_opencode_session(upstream: dict, client_value: str | None) -> str | None:
    if upstream.get("opencode") is False:
        return None
    if upstream.get("opencode") is not True and not is_opencode_endpoint(upstream.get("base_url", "")):
        return None
    if client_value:
        return client_value
    for k, v in (upstream.get("headers") or {}).items():
        if k.lower() == OPENCODE_SESSION_HEADER and str(v or "").strip():
            return None
    return str(upstream.get("session_id") or upstream.get("opencode_session") or _AUTO_SESSION_ID)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a JSON object")
    return cfg


def build_config(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    server = dict(cfg.get("server", {}))
    access = dict(cfg.get("access", {}))
    routing = dict(cfg.get("routing", {}))
    upstreams = list(cfg.get("upstreams", cfg.get("providers", [])))

    server["host"] = args.host or os.getenv("CLIPROXY_HOST", server.get("host", "127.0.0.1"))
    server["port"] = args.port or int(os.getenv("CLIPROXY_PORT", server.get("port", DEFAULT_PORT)))

    env_key = os.getenv("CLIPROXY_API_KEY")
    api_keys = list(access.get("api_keys", access.get("api-keys", [])))
    if args.api_key:
        api_keys = [args.api_key]
    elif env_key and not api_keys:
        api_keys = [env_key]
    access["api_keys"] = api_keys

    if args.upstream:
        upstreams = [{
            "name": "default",
            "base_url": args.upstream.rstrip("/"),
            "api_key": args.upstream_key or os.getenv("CLIPROXY_UPSTREAM_KEY", ""),
            "models": ["*"],
        }]

    routing.setdefault("strategy", "round-robin")
    routing.setdefault("retry", int(os.getenv("CLIPROXY_RETRY", 1)))
    routing.setdefault("cooldown_seconds", 60)

    return {"server": server, "access": access, "routing": routing, "upstreams": upstreams}


class UpstreamPool:
    def __init__(self, upstreams: list[dict], routing: dict):
        self._upstreams = [self._normalize(u, i) for i, u in enumerate(upstreams)]
        self._routing = routing
        self._lock = threading.Lock()
        self._rr_counter = itertools.count()
        self._cooldown_until: dict[str, float] = {}

    @staticmethod
    def _normalize(u: dict, i: int) -> dict:
        u = dict(u)
        u.setdefault("name", f"upstream-{i}")
        u["base_url"] = u.get("base_url", "").rstrip("/")
        u.setdefault("models", ["*"])
        u.setdefault("timeout", 120)
        if not u.get("api_key") and is_opencode_endpoint(u.get("base_url", "")):
            u["api_key"] = os.getenv("OPENCODE_API_KEY", "")
        return u

    @property
    def all(self) -> list[dict]:
        return list(self._upstreams)

    def mark_failure(self, name: str):
        with self._lock:
            self._cooldown_until[name] = time.time() + max(0, int(self._routing.get("cooldown_seconds", 60)))

    def eligible(self, model: str | None) -> list[dict]:
        now = time.time()
        out = []
        for u in self._upstreams:
            if self._cooldown_until.get(u["name"], 0) > now:
                continue
            patterns = u.get("models", ["*"])
            if not patterns or "*" in patterns:
                out.append(u)
            elif model and any(fnmatch.fnmatch(model, p) for p in patterns):
                out.append(u)
        return out

    def ordered(self, model: str | None) -> list[dict]:
        cands = self.eligible(model)
        if self._routing.get("strategy") == "fill-first" or len(cands) <= 1:
            return cands
        with self._lock:
            start = next(self._rr_counter) % len(cands)
        return cands[start:] + cands[:start]


def read_json_body(handler: BaseHTTPRequestHandler, limit: int = 32 * 1024 * 1024) -> tuple[bytes, dict]:
    try:
        n = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        n = 0
    if n > limit:
        return b"", {}
    raw = handler.rfile.read(n) if n > 0 else b""
    if not raw:
        return raw, {}
    try:
        return raw, json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return raw, {}


def send_json(handler: BaseHTTPRequestHandler, status: int, obj: dict):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def upstream_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if path.startswith("/v1/") and base.endswith("/v1"):
        return base + path[3:]
    return base + path


def build_upstream_request(upstream: dict, method: str, path: str, body: bytes,
                           content_type: str, session: str | None, accept: str,
                           ) -> urllib.request.Request:
    req = urllib.request.Request(upstream_url(upstream["base_url"], path),
                                 data=body if method in ("POST", "PUT", "PATCH") else None,
                                 method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    req.add_header("Accept", accept)
    if upstream.get("api_key"):
        req.add_header("Authorization", f"******{upstream['api_key']}")
    for k, v in (upstream.get("headers") or {}).items():
        req.add_header(k, v)
    resolved = resolve_opencode_session(upstream, session)
    if resolved:
        req.add_unredirected_header(OPENCODE_SESSION_HEADER, resolved)
    return req


def upstream_request(upstream: dict, method: str, path: str, body: bytes,
                     content_type: str, timeout: int,
                     session: str | None = None) -> tuple[int, list[tuple[str, str]], bytes]:
    req = build_upstream_request(upstream, method, path, body, content_type, session,
                                 "application/json, text/event-stream")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, list(resp.getheaders()), resp.read()
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read()
        except Exception:
            err_body = b""
        return e.code, list((e.headers or {}).items()), err_body


class Handler(BaseHTTPRequestHandler):
    pool: UpstreamPool
    config: dict
    server_version = f"cliproxy-py/{VERSION}"

    POST_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses")

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.command} {self.path} {fmt % args}\n")

    def _cors_preflight(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def _authorized(self) -> bool:
        keys = self.config.get("access", {}).get("api_keys", [])
        if not keys:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip() in keys
        return self.headers.get("X-API-Key", "").strip() in keys

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        send_json(self, 401, {"error": {"message": "invalid api key", "type": "auth_error", "code": 401}})
        return False

    def do_OPTIONS(self):
        self._cors_preflight()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            return send_json(self, 200, {"status": "ok"})
        if path == "/":
            return send_json(self, 200, {
                "message": "cliproxy-py server",
                "version": VERSION,
                "endpoints": [f"POST {p}" for p in self.POST_PATHS] + ["GET /v1/models"],
            })
        if path in ("/v1/models", "/models"):
            if not self._require_auth():
                return
            return self._handle_models()
        return send_json(self, 404, {"error": {"message": f"not found: {path}", "code": 404}})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in self.POST_PATHS:
            return send_json(self, 404, {"error": {"message": f"not found: {path}", "code": 404}})
        if not self._require_auth():
            return
        raw, data = read_json_body(self)
        model = data.get("model") if isinstance(data, dict) else None
        stream = bool(data.get("stream")) if isinstance(data, dict) else False
        session = (self.headers.get(OPENCODE_SESSION_HEADER) or "").strip() or None
        if stream and path != "/v1/embeddings":
            return self._handle_stream(path, raw, model, session)
        return self._handle_proxy(path, raw, model, session)

    def _handle_models(self):
        models: dict[str, str] = {}
        for u in self.pool.all:
            for p in u.get("models", ["*"]):
                if p != "*" and "*" not in p and p not in models:
                    models[p] = u["name"]
        for u in self.pool.all:
            try:
                status, _, body = upstream_request(u, "GET", "/models", b"", "", 15)
                if status == 200:
                    for m in json.loads(body.decode("utf-8")).get("data", []):
                        if m.get("id") and m["id"] not in models:
                            models[m["id"]] = u["name"]
            except Exception:
                continue
        send_json(self, 200, {"object": "list",
                              "data": [{"id": k, "object": "model", "owned_by": v}
                                       for k, v in sorted(models.items())]})

    def _candidates(self, model: str | None, message: str) -> list[dict] | None:
        cands = self.pool.ordered(model)
        if not cands:
            send_json(self, 503, {"error": {"message": message, "type": "upstream_error", "code": 503}})
            return None
        return cands

    def _try_upstreams(self, path: str, raw: bytes, model: str | None,
                       content_type: str, session: str | None
                       ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        cands = self._candidates(model, f"no upstream available for model '{model}' "
                                       "(all cooling down or no match)")
        if cands is None:
            return None
        retries = max(0, int(self.config.get("routing", {}).get("retry", 1)))
        last: tuple[int, list[tuple[str, str]], bytes] = (502, [], b"")
        for u in cands * (1 + retries):
            try:
                status, headers, body = upstream_request(
                    u, "POST", path, raw, content_type, int(u.get("timeout", 120)), session)
            except Exception as e:
                self.pool.mark_failure(u["name"])
                last = (502, [], json.dumps(
                    {"error": {"message": f"upstream {u['name']} unreachable: {e}",
                               "type": "upstream_error"}}).encode())
                continue
            if status in (408, 429, 500, 502, 503, 504, 401, 403):
                self.pool.mark_failure(u["name"])
            elif status != 404:
                return status, headers, body
            last = (status, headers, body)
        status, headers, body = last
        self._relay(status, headers, body or b'{"error":{"message":"all upstreams failed"}}')
        return None

    def _relay(self, status: int, headers: list[tuple[str, str]], body: bytes):
        ctype = next((v for k, v in headers if k.lower() == "content-type"), "application/json")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _handle_proxy(self, path: str, raw: bytes, model: str | None, session: str | None):
        res = self._try_upstreams(path, raw, model, self.headers.get("Content-Type", "application/json"), session)
        if res is None:
            return
        self._relay(*res)

    def _handle_stream(self, path: str, raw: bytes, model: str | None, session: str | None):
        cands = self._candidates(model, "no upstream available")
        if cands is None:
            return
        ctype = self.headers.get("Content-Type", "application/json")
        last_err, last_status = b'{"error":{"message":"all upstreams failed"}}', 502
        for u in cands:
            req = build_upstream_request(u, "POST", path, raw, ctype, session, "text/event-stream")
            try:
                resp = urllib.request.urlopen(req, timeout=int(u.get("timeout", 120)))
            except urllib.error.HTTPError as e:
                try:
                    last_err = e.read() or b""
                except Exception:
                    last_err = b""
                last_status = e.code
                if e.code == 404:
                    continue
                if e.code in (408, 429, 500, 502, 503, 504, 401, 403):
                    self.pool.mark_failure(u["name"])
                    continue
                break
            except Exception as e:
                self.pool.mark_failure(u["name"])
                last_err = json.dumps({"error": {"message": str(e)}}).encode()
                continue
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                resp.close()
                self.close_connection = True
                return
            except Exception:
                try:
                    resp.close()
                except Exception:
                    pass
                self.close_connection = True
                return
        try:
            self._relay(last_status, [("Content-Type", "application/json")], last_err)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cliproxy-py — lightweight OpenAI-compatible proxy")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--upstream", default=None)
    ap.add_argument("--upstream-key", default=None)
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0

    try:
        cfg = build_config(args)
    except (OSError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if not cfg["upstreams"]:
        print("warning: no upstreams configured — proxy returns 503 until you add one.", file=sys.stderr)
        print("hint: copy config.example.json to config.json, or pass --upstream URL --upstream-key KEY",
              file=sys.stderr)

    Handler.config = cfg
    Handler.pool = UpstreamPool(cfg["upstreams"], cfg["routing"])
    ThreadingHTTPServer.allow_reuse_address = True
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"bind {host}:{port} failed: {e}", file=sys.stderr)
        return 1
    auth_state = "open" if not cfg["access"]["api_keys"] else f"{len(cfg['access']['api_keys'])} key(s)"
    print(f"cliproxy-py v{VERSION} on http://{host}:{httpd.server_address[1]} "
          f"[{len(cfg['upstreams'])} upstream(s), auth={auth_state}]", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
