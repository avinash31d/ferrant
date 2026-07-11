import asyncio
import os
from ferrant import Agent, Team

async def main() -> None:
    key = os.environ["OPENAI_API_KEY"]
    researcher = Agent.openai("gpt-5-nano", key, instructions="Identify facts and uncertainty.")
    reviewer = Agent.openai("gpt-5-nano", key, instructions="Find risks and edge cases.")
    team = Team.openai("gpt-5-nano", key, [
        ("researcher", "Researches the subject", researcher),
        ("reviewer", "Critically reviews proposals", reviewer),
    ])
    print(await team.run("Propose a safe VM-to-containers migration and consult both specialists."))

asyncio.run(main())
