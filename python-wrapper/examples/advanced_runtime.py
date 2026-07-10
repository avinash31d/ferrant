import asyncio
import os
from liteagent import Agent, Retriever

async def main() -> None:
    retriever = Retriever(path=".liteagent/vectors.json")
    await retriever.upsert([
        {"id": "graph-guide", "text": "Graph checkpoints make workflow recovery durable.", "metadata": {"source": "guide"}},
        {"id": "trace-guide", "text": "Tracing and usage records make agent runs observable.", "metadata": {"source": "guide"}},
    ])
    print(await retriever.retrieve("durable recovery", limit=2))
    agent = Agent.openai("gpt-5-nano", os.environ["OPENAI_API_KEY"])
    await agent.run_stream("Explain durable agent workflows briefly.", print)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    print(await agent.run_structured("Summarize durable workflows.", schema))

asyncio.run(main())
