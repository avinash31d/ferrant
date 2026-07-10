# liteagent Python wrapper

This directory is an optional, thin Python binding around the Rust `liteagent`
crate. The agent loop, providers, MCP, graph scheduler, persistence, streaming,
and retrieval remain implemented in Rust. Installing this package does not
change the Rust crate or its examples.

```bash
pip install liteagent
```

Then import it directly:

```python
import asyncio
import os
from liteagent import Agent

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

The exposed surface covers OpenAI-compatible and Anthropic agents, Python
tools, MCP-discovered tools, coordinator teams, multimodal input/output,
streaming callbacks, schema-validated output, persistent retrieval, and durable
workflow graphs with routes, parallel joins, retry/timeout policies,
interrupts, resume, and recovery.
