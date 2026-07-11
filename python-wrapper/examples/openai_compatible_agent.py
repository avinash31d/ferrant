import asyncio
import os
from ferrant import Agent

async def main() -> None:
    agent = Agent.openai(
        os.getenv("OPENAI_COMPATIBLE_MODEL", "LiquidAI/LFM2.5-230M-GGUF:Q8_0"),
        os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:8080/v1"),
        instructions="Be concise and helpful.")
    print(await agent.run("In one sentence, explain why Go is memory safe."))

asyncio.run(main())
