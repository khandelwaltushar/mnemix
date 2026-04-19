"""Run the Mnemix proxy on http://localhost:8000 with defaults.

Usage:
    # 1. Install the dev server (not a hard dep of the library):
    pip install uvicorn

    # 2. Point Mnemix at your OpenAI key:
    export OPENAI_API_KEY=sk-...

    # 3. Start the proxy:
    python examples/basic_proxy.py

    # 4. In another shell, hit it twice with the same prompt:
    curl -si http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -d '{"model":"gpt-4o-mini","messages":[
              {"role":"user","content":"Summarize the plot of Hamlet"}
            ]}'

    The first call returns ``X-Mnemix-Cache: MISS`` and forwards upstream.
    The second identical call returns ``X-Mnemix-Cache: HIT`` with a
    sub-20ms latency — no upstream round-trip, no tokens billed.
"""

from __future__ import annotations

import sys

from mnemix.proxy import create_app
from mnemix.types import CacheConfig

HOST = "0.0.0.0"
PORT = 8000


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "error: this example needs uvicorn to run the FastAPI app.\n"
            "       install it first:\n\n"
            "           pip install uvicorn\n\n",
        )
        return 1

    config = CacheConfig()
    app = create_app(config)

    banner = f"""
{"=" * 70}
Mnemix semantic-cache proxy — listening on http://localhost:{PORT}
{"=" * 70}

Try it (OPENAI_API_KEY must be exported):

  curl -si http://localhost:{PORT}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer $OPENAI_API_KEY" \\
    -d '{{"model":"gpt-4o-mini","messages":[
           {{"role":"user","content":"Summarize the plot of Hamlet"}}
         ]}}'

  First call  → X-Mnemix-Cache: MISS   (forwarded to OpenAI, response cached)
  Second call → X-Mnemix-Cache: HIT    (served from cache, no upstream call)

Other endpoints:
  GET /health    liveness probe
  GET /metrics   hit-rate, tokens saved, estimated USD saved

Point an existing OpenAI SDK client at this proxy by changing only
base_url to http://localhost:{PORT}/v1 — no other code changes.

Press Ctrl+C to stop.
{"=" * 70}
"""
    print(banner)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
