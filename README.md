# cliproxy-py

Lightweight OpenAI-compatible proxy in **pure Python (stdlib only)** — an easy-use subset inspired by [router-for-me/CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) (Go).

Upstream CLIProxyAPI wraps CLI accounts (Antigravity, Codex, Claude Code, Grok, Muse, Devin, …) behind OpenAI / Gemini / Claude / Codex APIs. This Python version keeps the same idea but minimal:

- Single file: `cliproxy.py`, zero third-party deps
- OpenAI-compatible: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/responses`
- Multi-upstream with round-robin / fill-first, model-pattern routing, failover + cooldown
- **OpenCode Go provider**: auto-injects the required `x-opencode-session` routing header for `opencode.ai` endpoints (per [pi#10820](https://github.com/earendil-works/pi/pull/10820)); `OPENCODE_API_KEY` env fallback
- Multi-upstream with round-robin / fill-first, model-pattern routing, failover + cooldown
- SSE streaming passthrough
- Bearer-key auth for your clients, per-upstream keys
- JSON config + env vars + CLI flags

## Quick start

```powershell
cd cliproxy-py
copy config.example.json config.json
# edit config.json: set access.api_keys and upstreams[0].api_key
python cliproxy.py
# or without a config file:
python cliproxy.py --upstream https://api.openai.com/v1 --upstream-key sk-... --api-key my-local-key --port 8317
```

Point any OpenAI client at it:

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8317/v1"
$env:OPENAI_API_KEY = "my-local-key"
```

Test:

```powershell
curl http://127.0.0.1:8317/healthz
curl -H "Authorization: Bearer my-local-key" http://127.0.0.1:8317/v1/models
curl -H "Authorization: Bearer my-local-key" -H "Content-Type: application/json" `
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}' `
  http://127.0.0.1:8317/v1/chat/completions
```

## Config

`config.json` (see `config.example.json`):

```json
{
  "server": { "host": "127.0.0.1", "port": 8317 },
  "access": { "api_keys": ["your-api-key-1"] },
  "routing": { "strategy": "round-robin", "retry": 1, "cooldown_seconds": 60 },
  "upstreams": [
    { "name": "openai", "base_url": "https://api.openai.com/v1",
      "api_key": "sk-...", "models": ["*"], "timeout": 120 },
    { "name": "groq", "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-...", "models": ["llama-*", "mixtral-*"] }
  ]
}
```

- `models`: fnmatch patterns; `"*"` matches everything. A request for `model` only uses upstreams whose patterns match.
- `strategy`: `round-robin` (default) or `fill-first`.
- `retry`: extra full rounds over eligible upstreams on 408/429/5xx.
- Failed upstreams cool down for `cooldown_seconds` (transient + auth errors).

Env overrides: `CLIPROXY_HOST`, `CLIPROXY_PORT`, `CLIPROXY_API_KEY`, `CLIPROXY_UPSTREAM_KEY`, `CLIPROXY_RETRY`.

CLI flags: `--config`, `--host`, `--port`, `--api-key`, `--upstream`, `--upstream-key`. `--upstream` replaces the config list with a single upstream (easiest path).

Leave `access.api_keys` empty for open mode (no client auth — localhost only recommended).

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/healthz` | `{"status":"ok"}` |
| GET | `/` | service info |
| GET | `/v1/models` | aggregates upstream `/models` + configured patterns |
| POST | `/v1/chat/completions` | non-stream relay + SSE failover |
| POST | `/v1/completions` | legacy completions relay |
| POST | `/v1/embeddings` | relay, never treated as stream |
| POST | `/v1/responses` | responses API relay (e.g. OpenCode Go) |

Error translation is pass-through: upstream status + body are relayed; `401` with a bad client key; `503` when no upstream is eligible.

## OpenCode Go provider

[OpenCode Go](https://opencode.ai) exposes OpenAI-compatible chat + responses
APIs at `https://opencode.ai/zen/go/v1` (model discovery at
`https://opencode.ai/inference/go/v1`). Its zen endpoints reject requests
without a per-conversation routing header (`400 MissingSessionID`), so the
proxy injects `x-opencode-session` for any upstream served by `opencode.ai`
(hostname match, per [pi#10820](https://github.com/earendil-works/pi/pull/10820)):

Precedence: your downstream `x-opencode-session` header (any casing wins) →
per-upstream `session_id` in config → stable auto-generated id per proxy
process (so requests just work out of the box). Set `"opencode": true` to
force injection for a non-`opencode.ai` mirror, or `"opencode": false` to opt
out. Like pi's providers, an opencode endpoint with no `api_key` picks up
`OPENCODE_API_KEY` from the environment.

```json
{ "name": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1",
  "api_key": "sk-...", "models": ["kimi-k2.7-code", "deepseek-*"],
  "session_id": "my-conversation-id", "opencode": true }
```

Use `session_id` per conversation (oh-my-pi also uses it as a prompt-cache
affinity handle on Go lanes) so the gateway routes and caches consistently.

## Compared to upstream CLIProxyAPI

| | CLIProxyAPI (Go) | cliproxy-py |
|---|---|---|
| Protocols | OpenAI/Responses, Claude, Gemini, Codex realtime, … | OpenAI chat/completions/embeddings/responses |
| Auth sources | OAuth login flows, credential files, multi-account rotation | Static per-upstream API keys |
| Routing | session affinity, weights, cooldown files, mDNS | round-robin / fill-first + cooldown |
| Extras | TUI, management API, plugins, Redis queue | none — one file, stdlib only |

To grow toward upstream later: add `/v1/messages`, `/v1beta/models/*action` translators and an auth-manager per provider.

## Smoke test

```powershell
python test_smoke.py
```

Spins up two fake OpenAI upstreams on localhost, starts the proxy, and checks health, auth, models aggregation, chat relay, failover, and SSE streaming.
