"""
cliproxy-py — lightweight OpenAI-compatible proxy (Python stdlib only).

Inspired by router-for-me/CLIProxyAPI (Go). This is a minimal,
easy-use subset focusing on OpenAI compatibility:

  GET  /healthz                health check
  GET  /                       service info
  GET  /v1/models              aggregated model list
  POST /v1/chat/completions    chat (stream + non-stream passthrough)
  POST /v1/completions         legacy completions passthrough
  POST /v1/embeddings          embeddings passthrough
  POST /v1/responses           responses API passthrough (e.g. OpenCode Go)

  OpenCode Go (https://opencode.ai/zen/go/v1) is OpenAI-compatible: point an
  upstream at it and the proxy injects the required `x-opencode-session`
  routing header (see earendil-works/pi#10820).

Features:
  - Zero third-party dependencies (stdlib only)
  - Multi-upstream with round-robin / fill-first + failover + cooldown
  - Bearer auth for downstream clients, per-upstream keys
  - SSE streaming passthrough
  - JSON config file + env vars + CLI flags

Quick start:
  copy config.example.json config.json   # edit keys
  python cliproxy.py
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

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}

# ----------------------------------------------- opencode go provider ---
# OpenCode's zen endpoints (opencode.ai, e.g. https://opencode.ai/zen/go/v1)
# reject inference requests lacking a per-conversation routing header with
# `400 MissingSessionID`. Per earendil-works/pi#10820 the injection is keyed
# off the endpoint hostname — not a provider id — so custom routes pointed at
# OpenCode behave like the built-in providers. Caller-provided values (any
# case) always win; nothing is fabricated for non-OpenCode upstreams.

OPENCODE_SESSION_HEADER = "x-opencode-session"
OPENCODE_HOST = "opencode.ai"

# Stable per-process id so out-of-the-box requests don't 400; override per
# upstream with "session_id" (or send your own downstream header per call).
_AUTO_SESSION_ID = uuid.uuid4().hex


def is_opencode_endpoint(base_url: str | None) -> bool:
    """True when a base URL is served by OpenCode Zen / OpenCode Go."""
    if not base_url:
        return False
    try:
        return urlparse(base_url).hostname == OPENCODE_HOST
    except Exception:
        return False


def resolve_opencode_session(upstream: dict, client_value: str | None) -> str | None:
    """Value to send as `x-opencode-session`, or None when not applicable."""
    if upstream.get("opencode") is False:  # explicit opt-out
        return None
    if upstream.get("opencode") is not True and not is_opencode_endpoint(upstream.get("base_url", "")):
        return None
    if client_value:
        return client_value  # caller's downstream header wins
    for k, v in (upstream.get("headers") or {}).items():
        if k.lower() == OPENCODE_SESSION_HEADER and str(v or "").strip():
            return None  # explicit per-upstream header already covers it
    return str(upstream.get("session_id") or upstream.get("opencode_session")
               or _AUTO_SESSION_ID)

# ---------------------------------------------------------------- config ---


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

    # env overrides (easy-use: CLIPROXY_PORT, CLIPROXY_API_KEY, CLIPROXY_UPSTREAM etc.)
    server["host"] = args.host or os.getenv("CLIPROXY_HOST", server.get("host", "127.0.0.1"))
    server["port"] = args.port or int(os.getenv("CLIPROXY_PORT", server.get("port", DEFAULT_PORT)))

    env_key = os.getenv("CLIPROXY_API_KEY")
    api_keys = list(access.get("api_keys", access.get("api-keys", [])))
    if args.api_key:
        api_keys = [args.api_key]
    elif env_key and not api_keys:
        api_keys = [env_key]
    access["api_keys"] = api_keys

    # Single-upstream shortcut: --upstream https://... --upstream-key sk-...
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

    return {
        "server": server,
        "access": access,
        "routing": routing,
        "upstreams": upstreams,
    }


# --------------------------------------------------------------- routing ---

class UpstreamPool:
    """Thread-safe round-robin / fill-first selection with cooldown."""

    def __init__(self, upstreams: list[dict], routing: dict):
        self._lock = threading.Lock()
        self._upstreams = [self._normalize(u, i) for i, u in enumerate(upstreams)]
        self._routing = routing
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
            # Same convention as pi's opencode providers: OPENCODE_API_KEY env.
            u["api_key"] = os.getenv("OPENCODE_API_KEY", "")
        return u

    def _in_cooldown(self, name: str) -> bool:
        return self._cooldown_until.get(name, 0) > time.time()

    def mark_failure(self, name: str, transient: bool, cooldown_seconds: int = 60):
        if transient:
            with self._lock:
                self._cooldown_until[name] = time.time() + max(0, cooldown_seconds)

    def eligible(self, model: str | None) -> list[dict]:
        out = []
        for u in self._upstreams:
            if self._in_cooldown(u["name"]):
                continue
            patterns = u.get("models", ["*"])
            if not patterns or "*" in patterns:
                out.append(u)
                continue
            if model and any(fnmatch.fnmatch(model, p) for p in patterns):
                out.append(u)
        return out

    def ordered(self, model: str | None) -> list[dict]:
        cands = self.eligible(model)
        strategy = self._routing.get("strategy", "round-robin")
        if strategy == "fill-first" or len(cands) <= 1:
            return cands
        # round-robin rotation
        with self._lock:
            start = next(self._rr_counter) % len(cands)
        return cands[start:] + cands[:start]


# ------------------------------------------------------------------ http ---

def read_json_body(handler: BaseHTTPRequestHandler, limit: int = 32 * 1024 * 1024) -> tuple[bytes, dict]:
    length = handler.headers.get("Content-Length")
    try:
        n = int(length) if length else 0
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
    """Join base URL and path without duplicating a trailing /v1."""
    base = base_url.rstrip("/")
    if path.startswith("/v1/") and base.endswith("/v1"):
        return base + path[3:]
    return base + path


def upstream_request(upstream: dict, method: str, path: str, body: bytes,
                     content_type: str, timeout: int,
                     session: str | None = None) -> tuple[int, list[tuple[str, str]], bytes]:
    """Non-streaming upstream call. Returns (status, headers, body)."""
    url = upstream_url(upstream["base_url"], path)
    req = urllib.request.Request(url, data=body if method in ("POST", "PUT", "PATCH") else None, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    req.add_header("Accept", "application/json, text/event-stream")
    if upstream.get("api_key"):
        req.add_header("Authorization", f"Bearer {upstream['api_key']}")
    for k, v in (upstream.get("headers") or {}).items():
        req.add_header(k, v)
    resolved = resolve_opencode_session(upstream, session)
    if resolved:
        # add_unredirected_header avoids urllib stripping it on redirects
        req.add_unredirected_header(OPENCODE_SESSION_HEADER, resolved)
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

    # -- helpers -------------------------------------------------
    def log_message(self, fmt, *args):  # quieter, single-line logs
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
            return True  # open mode when no keys configured
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip() in keys
        return self.headers.get("X-API-Key", "").strip() in keys

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        send_json(self, 401, {"error": {"message": "invalid api key", "type": "auth_error", "code": 401}})
        return False

    # -- routing -------------------------------------------------
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
                "endpoints": ["POST /v1/chat/completions", "POST /v1/completions",
                                    "POST /v1/embeddings", "POST /v1/responses",
                                    "GET /v1/models"],
            })
        if path in ("/v1/models", "/models"):
            if not self._require_auth():
                return
            return self._handle_models()
        return send_json(self, 404, {"error": {"message": f"not found: {path}", "code": 404}})

    def _client_session_header(self) -> str | None:
        for k, v in (self.headers.items() if hasattr(self.headers, "items") else []):
            if k.lower() == OPENCODE_SESSION_HEADER and str(v or "").strip():
                return str(v).strip()
        return None

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings",
                    "/v1/responses"):
            if not self._require_auth():
                return
            raw, data = read_json_body(self)
            model = data.get("model") if isinstance(data, dict) else None
            stream = bool(data.get("stream")) if isinstance(data, dict) else False
            session = self._client_session_header()
            if stream and path != "/v1/embeddings":
                return self._handle_stream(path, raw, model, session)
            return self._handle_proxy(path, raw, model, session)
        return send_json(self, 404, {"error": {"message": f"not found: {path}", "code": 404}})

    # -- handlers ------------------------------------------------
    def _handle_models(self):
        """Aggregate upstream /v1/models; fall back to configured patterns."""
        models: dict[str, str] = {}
        for u in self.pool._upstreams:  # noqa: SLF001 (same class family)
            for p in u.get("models", ["*"]):
                if p != "*" and "*" not in p and p not in models:
                    models[p] = u["name"]
        # try live upstream listing
        for u in self.pool._upstreams:
            try:
                status, _, body = upstream_request(u, "GET", "/models", b"", "", 15)
                if status == 200:
                    payload = json.loads(body.decode("utf-8"))
                    for m in payload.get("data", []):
                        mid = m.get("id")
                        if mid and mid not in models:
                            models[mid] = u["name"]
            except Exception:
                continue
        data = [{"id": k, "object": "model", "owned_by": v} for k, v in sorted(models.items())]
        send_json(self, 200, {"object": "list", "data": data})

    def _try_upstreams(self, path: str, raw: bytes, model: str | None,
                       content_type: str,
                       session: str | None = None) -> tuple[int, list[tuple[str, str]], bytes] | None:
        routing = self.config.get("routing", {})
        retries = max(0, int(routing.get("retry", 1)))
        cooldown = int(routing.get("cooldown_seconds", 60))
        cands = self.pool.ordered(model)
        if not cands:
            send_json(self, 503, {"error": {"message": "no upstream available for model "
                                                       f"'{model}' (all cooling down or no match)",
                                            "type": "upstream_error", "code": 503}})
            return None
        attempts = min(len(cands) * (1 + retries), max(len(cands), 1))
        last: tuple[int, list[tuple[str, str]], bytes] | None = (502, [], b"")
        tried = 0
        for u in (cands * (1 + retries))[:attempts]:
            tried += 1
            try:
                status, headers, body = upstream_request(
                    u, "POST", path, raw, content_type, int(u.get("timeout", 120)), session)
            except Exception as e:  # connection error → transient
                self.pool.mark_failure(u["name"], True, cooldown)
                last = (502, [], json.dumps(
                    {"error": {"message": f"upstream {u['name']} unreachable: {e}",
                               "type": "upstream_error"}}).encode())
                continue
            if status in (408, 429, 500, 502, 503, 504):
                transient = status != 429 or tried < attempts
                self.pool.mark_failure(u["name"], True, cooldown)
                last = (status, headers, body)
                continue
            if status == 404:
                # model not on this upstream — try next without cooling down
                last = (status, headers, body)
                continue
            if status in (401, 403):
                self.pool.mark_failure(u["name"], True, cooldown)
                last = (status, headers, body)
                continue
            return status, headers, body
        # exhausted → relay last error verbatim when possible
        status, headers, body = last
        self._relay(status, headers, body or b'{"error":{"message":"all upstreams failed"}}')
        return None

    def _relay(self, status: int, headers: list[tuple[str, str]], body: bytes):
        ctype = "application/json"
        for k, v in headers:
            if k.lower() == "content-type":
                ctype = v
                break
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _handle_proxy(self, path: str, raw: bytes, model: str | None,
                      session: str | None = None):
        ctype = self.headers.get("Content-Type", "application/json")
        res = self._try_upstreams(path, raw, model, ctype, session)
        if res is None:
            return  # error already sent
        status, headers, body = res
        self._relay(status, headers, body)

    def _handle_stream(self, path: str, raw: bytes, model: str | None,
                       session: str | None = None):
        """Streaming SSE passthrough with failover on connection setup."""
        routing = self.config.get("routing", {})
        cooldown = int(routing.get("cooldown_seconds", 60))
        cands = self.pool.ordered(model)
        if not cands:
            return send_json(self, 503, {"error": {"message": "no upstream available", "code": 503}})
        ctype = self.headers.get("Content-Type", "application/json")
        last_err: bytes = b""
        last_status = 502
        for u in cands:
            url = upstream_url(u["base_url"], path)
            req = urllib.request.Request(url, data=raw, method="POST")
            req.add_header("Content-Type", ctype)
            req.add_header("Accept", "text/event-stream")
            if u.get("api_key"):
                req.add_header("Authorization", f"Bearer {u['api_key']}")
            for k, v in (u.get("headers") or {}).items():
                req.add_header(k, v)
            resolved = resolve_opencode_session(u, session)
            if resolved:
                req.add_unredirected_header(OPENCODE_SESSION_HEADER, resolved)
            try:
                resp = urllib.request.urlopen(req, timeout=int(u.get("timeout", 120)))
            except urllib.error.HTTPError as e:
                try:
                    last_err = e.read() or b""
                except Exception:
                    last_err = b""
                last_status = e.code
                if e.code == 404:
                    continue  # model not on this upstream; try next
                if e.code in (408, 429, 500, 502, 503, 504, 401, 403):
                    self.pool.mark_failure(u["name"], True, cooldown)
                    continue
                break
            except Exception as e:
                self.pool.mark_failure(u["name"], True, cooldown)
                last_err = json.dumps({"error": {"message": str(e)}}).encode()
                continue
            # connected — relay chunks verbatim (close-delimited body)
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
        # all failed → JSON error (non-SSE so clients surface it)
        try:
            body = last_err or b'{"error":{"message":"all upstreams failed"}}'
            self._relay(last_status, [("Content-Type", "application/json")], body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cliproxy-py — lightweight OpenAI-compatible proxy")
    ap.add_argument("--config", default="config.json", help="path to JSON config file")
    ap.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="bind port")
    ap.add_argument("--api-key", default=None, help="downstream client key (overrides config)")
    ap.add_argument("--upstream", default=None, help="single upstream base URL shortcut")
    ap.add_argument("--upstream-key", default=None, help="single upstream api key")
    ap.add_argument("--version", action="store_true", help="print version and exit")
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
        print("warning: no upstreams configured — /v1/models will be empty and "
              "proxy calls return 503 until you add one.", file=sys.stderr)
        print("hint: copy config.example.json to config.json, or pass "
              "--upstream https://api.openai.com/v1 --upstream-key sk-...", file=sys.stderr)

    Handler.config = cfg
    Handler.pool = UpstreamPool(cfg["upstreams"], cfg["routing"])
    # silence deprecation: allow address reuse
    ThreadingHTTPServer.allow_reuse_address = True
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"bind {host}:{port} failed: {e}", file=sys.stderr)
        return 1
    auth_state = "open" if not cfg["access"]["api_keys"] else f"{len(cfg['access']['api_keys'])} key(s)"
    actual_port = httpd.server_address[1]
    print(f"cliproxy-py v{VERSION} on http://{host}:{actual_port} "
          f"[{len(cfg['upstreams'])} upstream(s), auth={auth_state}]", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
