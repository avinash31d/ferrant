![Ferrant Logo](../assets/logos.png)
# ferrant Python wrapper

This directory is an optional, thin Python binding around the Rust `ferrant`
crate. The agent loop, providers, MCP, graph scheduler, persistence, streaming,
and retrieval remain implemented in Rust. Installing this package does not
change the Rust crate or its examples.

```bash
pip install ferrant
```

Then import it directly:

```python
import asyncio
import os
from ferrant import Agent

async def main():
    agent = Agent.openai("gpt-5-nano", os.environ["OPENAI_API_KEY"])
    print(await agent.run("Explain Rust ownership in one paragraph."))

asyncio.run(main())
```

For local wrapper development, use `maturin develop --release` from this
directory. End users do not need Maturin or a Rust toolchain when installing a
prebuilt wheel.

The wheel uses PyO3's stable ABI for Python 3.9+. Rust futures are exposed as
normal `asyncio` awaitables. Python custom-tool and graph-node callbacks are
synchronous by design; keep expensive execution in Rust tools, MCP servers, or
model calls.

See `examples/` for Python counterparts of every top-level Rust example and
matched advanced workflow examples.

## Focused Python examples

Run these from the `python-wrapper/` directory after installing the package:

```bash
python examples/streaming.py
python examples/memory.py
python examples/rag.py
DOCUMENT_PATH=invoice.pdf python examples/document_extraction.py
DOCUMENT_PATH=invoice.pdf python examples/rag_document_extraction.py
```

- `streaming.py` prints `content_delta` events as the model generates them.
- `memory.py` stores a session in `.ferrant/sessions` and recalls a prior turn.
- `rag.py` persists a local hybrid vector index, retrieves relevant documents,
  and provides the matches as grounded agent context.
- `document_extraction.py` base64-encodes a local PDF and asks OpenAI to
  extract invoice fields. Set `DOCUMENT_PATH` to a PDF and
  `OPENAI_API_KEY` before running it.
- `rag_document_extraction.py` transcribes a local PDF with OpenAI, indexes the
  text locally, retrieves invoice-relevant passages, and returns
  schema-validated extraction results. Set `DOCUMENT_PATH` to a PDF.

The exposed surface covers OpenAI-compatible and Anthropic agents, Python
tools, MCP-discovered tools, coordinator teams, multimodal input/output,
streaming callbacks, schema-validated output, persistent retrieval, and durable
workflow graphs with routes, parallel joins, retry/timeout policies,
interrupts, resume, and recovery.
