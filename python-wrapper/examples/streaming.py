"""Print an OpenAI response as text arrives."""
import asyncio
import os

from ferrant import Agent


def print_event(event: dict) -> None:
    """Only print incremental text; tool and usage events remain available too."""
    if event["type"] == "content_delta":
        print(event["delta"], end="", flush=True)


async def main() -> None:
    agent = Agent.openai(
        "gpt-5-nano",
        os.environ["OPENAI_API_KEY"],
        instructions="Be concise and helpful.",
    )
    response = await agent.run_stream(
        "Explain why streaming improves chat UX in two sentences.",
        print_event,
    )
    print("\n\nFinal response:", response["content"])


asyncio.run(main())
