import asyncio
import os
from ferragent import Agent

async def main() -> None:
    agent = Agent.openai("gpt-5.4-mini", os.environ["OPENAI_API_KEY"],
        instructions="Describe visual evidence and uncertainty precisely.")
    response = await agent.run_multimodal([
        {"type": "text", "text": "What is shown in this image?"},
        {"type": "image_url", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"},
    ])
    print(response["content"])

asyncio.run(main())
