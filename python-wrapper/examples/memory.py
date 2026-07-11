"""Persist a conversation and recall it in a later turn."""
import asyncio
import os

from ferrant import Agent


async def main() -> None:
    agent = Agent.openai(
        "gpt-5-nano",
        os.environ["OPENAI_API_KEY"],
        instructions="Remember details from the current session and answer concisely.",
        storage_path=".ferrant/sessions",
    )

    session_id = "demo-user"
    print(await agent.run_session(session_id, "My favorite language is Rust."))
    print(await agent.run_session(session_id, "What is my favorite language?"))


asyncio.run(main())
