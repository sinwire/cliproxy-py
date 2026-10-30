"""Smoke test: fake upstreams + proxy, check health/auth/models/relay/failover/SSE."""
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def make_upstream(models, fail_chat_once=False):
    state = {"chat_calls": 0, "seen_session": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, status=200, ctype="application/json"):
            body = (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p.startswith("/v1/"):
                p = p[3:]
            if p == "/models":
                return self._send({"object": "list",
                                   "data": [{"id": m, "object": "model"} for m in models]})
            return self._send({"error": "nf"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            p = self.path.split("?", 1)[0]
            if p.startswith("/v1/"):
                p = p[3:]
            try:
                data = json.loads(raw.decode())
            except Exception:
                data = {}
            if p == "/chat/completions":
                state["chat_calls"] += 1
                state["seen_session"].append(self.headers.get("x-opencode-session"))
                if fail_chat_once and state["chat_calls"] == 1:
                    return self._send({"error": {"message": "busy"}}, 503)
                if data.get("stream"):
                    chunks = []
                    for tok in ["hel", "lo"]:
                        chunk = {"choices": [{"delta": {"content": tok}}]}
                        chunks.append(f"data: {json.dumps(chunk)}\n\n".encode())
                    chunks.append(b"data: [DONE]\n\n")
                    return self._send(b"".join(chunks), ctype="text/event-stream")
                return self._send({"choices": [{"message": {"content": "hello"}}]})
            if p == "/responses":
                state["seen_session"].append(self.headers.get("x-opencode-session"))
                return self._send({"output": [{"content": [{"text": "go-hello"}]}]})
            if p == "/embeddings":
                return self._send({"data": [{"embedding": [0.1]}]})
            return self._send({"error": "nf"}, 404)

    srv = ThreadingHTTPServer((HOST, 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.test_state = state
    return srv


def req(port, path, key=None, method="GET", payload=None, headers=None):
    url = f"http://{HOST}:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    if key:
        r.add_header("Authorization", f"Bearer {key}")
    if payload is not None:
        r.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read(), dict(resp.getheaders())
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {}


def main():
    up1 = make_upstream(["fake-a"], fail_chat_once=True)
    up2 = make_upstream(["fake-b"])
    p1, p2 = up1.server_address[1], up2.server_address[1]

    cfg = {
        "server": {"host": HOST, "port": 0},
        "access": {"api_keys": ["test-key"]},
        "routing": {"strategy": "round-robin", "retry": 1, "cooldown_seconds": 60},
        "upstreams": [
            {"name": "up1", "base_url": f"http://{HOST}:{p1}", "models": ["*"]},
            {"name": "up2", "base_url": f"http://{HOST}:{p2}", "models": ["*"]},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name

    proc = subprocess.Popen(
        [sys.executable, "cliproxy.py", "--config", cfg_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="cliproxy-py" if __import__("os").path.isdir("cliproxy-py") else ".",
    )
    proxy_port = None
    for _ in range(100):
        line = proc.stdout.readline() if proc.stdout else ""
        if "on http://" in line:
            try:
                proxy_port = int(line.rsplit(":", 1)[1].split()[0].rstrip(")"))
                break
            except ValueError:
                pass
        if proc.poll() is not None:
            print(proc.stdout.read() if proc.stdout else "")
            print("proxy failed to start")
            return 1
        time.sleep(0.05)
    # fallback: parse first line fully
    if proxy_port is None:
        print("could not detect proxy port")
        proc.terminate()
        return 1
    print(f"proxy on :{proxy_port}, upstreams :{p1} :{p2}")

    try:
        s, b, _ = req(proxy_port, "/healthz")
        check("healthz 200", s == 200 and json.loads(b)["status"] == "ok", f"got {s} {b!r}")

        s, b, _ = req(proxy_port, "/v1/models")
        check("models require auth", s == 401, f"got {s}")

        s, b, _ = req(proxy_port, "/v1/models", key="test-key")
        ids = sorted(m["id"] for m in json.loads(b).get("data", []))
        check("models aggregate fake-a+fake-b", ids == ["fake-a", "fake-b"], f"got {ids}")

        s, b, _ = req(proxy_port, "/v1/chat/completions", key="bad-key",
                       method="POST", payload={"model": "x", "messages": []})
        check("chat rejects bad key", s == 401, f"got {s}")

        # round-robin first call hits up1 which fails once -> failover to up2
        s, b, _ = req(proxy_port, "/v1/chat/completions", key="test-key",
                       method="POST", payload={"model": "fake-a",
                                               "messages": [{"role": "user", "content": "hi"}]})
        ok = s == 200 and "hello" in b.decode()
        check("chat failover up1->up2", ok, f"got {s} {b!r}")

        # SSE stream
        url = f"http://{HOST}:{proxy_port}/v1/chat/completions"
        r = urllib.request.Request(url, data=json.dumps(
            {"model": "m", "messages": [], "stream": True}).encode(), method="POST")
        r.add_header("Authorization", "Bearer test-key")
        r.add_header("Content-Type", "application/json")
        r.add_header("Accept", "text/event-stream")
        with urllib.request.urlopen(r, timeout=10) as resp:
            body = resp.read().decode()
        check("SSE contains chunks + DONE", "hel" in body and "[DONE]" in body, repr(body[:120]))

        s, b, _ = req(proxy_port, "/v1/embeddings", key="test-key",
                       method="POST", payload={"model": "m", "input": "hi"})
        check("embeddings relay", s == 200 and "embedding" in b.decode(), f"got {s} {b!r}")

        # --- opencode-go provider semantics (PR earendil-works/pi#10820) ---
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import cliproxy as _cx
        check("is_opencode_endpoint zen/go",
              _cx.is_opencode_endpoint("https://opencode.ai/zen/go/v1") is True)
        check("is_opencode_endpoint zen",
              _cx.is_opencode_endpoint("https://opencode.ai/zen/v1") is True)
        check("is_opencode_endpoint rejects other host",
              _cx.is_opencode_endpoint("https://api.openai.com/v1") is False)
        check("is_opencode_endpoint rejects junk/None",
              _cx.is_opencode_endpoint("not a url") is False
              and _cx.is_opencode_endpoint(None) is False)

        # opencode-flagged upstream against local fake: exercises injection path
        go_cfg = {
            "server": {"host": HOST, "port": 0},
            "access": {"api_keys": ["test-key"]},
            "routing": {"strategy": "fill-first", "retry": 0},
            "upstreams": [
                {"name": "go", "base_url": f"http://{HOST}:{p2}",
                 "models": ["*"], "opencode": True, "session_id": "sess-123"},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(go_cfg, f)
            go_cfg_path = f.name
        go_proc = subprocess.Popen(
            [sys.executable, "cliproxy.py", "--config", go_cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd="cliproxy-py" if _os.path.isdir("cliproxy-py") else ".",
        )
        go_port = None
        for _ in range(100):
            line = go_proc.stdout.readline() if go_proc.stdout else ""
            if "on http://" in line:
                try:
                    go_port = int(line.rsplit(":", 1)[1].split()[0].rstrip(")"))
                    break
                except ValueError:
                    pass
            if go_proc.poll() is not None:
                break
            time.sleep(0.05)
        try:
            check("go proxy started", go_port is not None)
            if go_port:
                up2.test_state["seen_session"].clear()
                s, b, _ = req(go_port, "/v1/chat/completions", key="test-key",
                               method="POST", payload={"model": "m", "messages": []})
                check("go chat relay", s == 200 and "hello" in b.decode(), f"got {s} {b!r}")
                check("go upstream got x-opencode-session sess-123",
                      up2.test_state["seen_session"][-1:] == ["sess-123"],
                      repr(up2.test_state["seen_session"][-1:]))

                # caller-provided downstream header wins
                up2.test_state["seen_session"].clear()
                s, _, _ = req(go_port, "/v1/chat/completions", key="test-key",
                               method="POST", payload={"model": "m", "messages": []},
                               headers={"x-opencode-session": "caller-1"})
                check("go caller session header wins",
                      up2.test_state["seen_session"][-1:] == ["caller-1"],
                      repr(up2.test_state["seen_session"][-1:]))

                # /v1/responses passthrough carries the header too
                up2.test_state["seen_session"].clear()
                s, b, _ = req(go_port, "/v1/responses", key="test-key",
                               method="POST", payload={"model": "m", "input": "hi"})
                check("go /v1/responses relay",
                      s == 200 and "go-hello" in b.decode(), f"got {s} {b!r}")
                check("go responses got session header",
                      up2.test_state["seen_session"][-1:] == ["sess-123"],
                      repr(up2.test_state["seen_session"][-1:]))

                # non-opencode upstream sends no session header
                up1.test_state["seen_session"].clear()
                s, _, _ = req(proxy_port, "/v1/chat/completions", key="test-key",
                               method="POST", payload={"model": "m", "messages": []})
                check("plain upstream gets no session header",
                      all(v is None for v in up1.test_state["seen_session"][-1:]),
                      repr(up1.test_state["seen_session"][-1:]))
        finally:
            go_proc.terminate()
    finally:
        proc.terminate()
        up1.shutdown()
        up2.shutdown()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
