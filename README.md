# Mnemix

**Stop paying for the same LLM call twice.** Mnemix intercepts your OpenAI/Anthropic calls, finds semantically similar cached responses, and returns them instantly — with configurable safety thresholds and full observability.

## Status

Alpha. Repo scaffolding only. See the [docs/](docs/) folder for design notes as they land.

## Install (dev)

```bash
pip install -e ".[dev]"
```

## Run the proxy

```bash
python examples/basic_proxy.py
```

## Layout

- [src/mnemix/](src/mnemix/) — library + proxy source
- [examples/](examples/) — runnable demos
- [tests/](tests/) — pytest suite (>=80% coverage)
- [docs/](docs/) — design notes

## License

MIT
