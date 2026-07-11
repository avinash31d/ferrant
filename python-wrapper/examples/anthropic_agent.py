import asyncio
import os
from ferragent import Agent, Tool

async def main() -> None:
    search = Tool("search_docs", "Search internal documentation", {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
    }, lambda args: f"Setup Guide, API Reference, and FAQ mention {args['query']}.")
    agent = Agent.anthropic("claude-sonnet-4-6", os.environ["ANTHROPIC_API_KEY"],
                            instructions="Search docs before answering.", tools=[search])
    print(await agent.run("How do I authenticate with the API?"))

asyncio.run(main())
