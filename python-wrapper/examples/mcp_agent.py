import asyncio
import os
from ferrant import Agent, McpTools

async def main() -> None:
    mcp = await McpTools.connect("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
    print("Discovered:", mcp.names())
    agent = Agent.openai("gpt-5-nano", os.environ["OPENAI_API_KEY"],
        instructions="Inspect files but never modify them unless asked.", mcp=mcp)
    print(await agent.run("Summarize this project's Rust source files."))

asyncio.run(main())
