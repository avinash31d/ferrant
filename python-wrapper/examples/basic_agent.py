import asyncio
import os
from ferragent import Agent, Tool

def weather(args: dict) -> str:
    return f"It's 22C and sunny in {args.get('city', 'unknown')}."

async def main() -> None:
    tool = Tool("get_weather", "Get weather for a city", {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]
    }, weather)
    agent = Agent.openai("gpt-5-nano", os.environ["OPENAI_API_KEY"],
                         instructions="Use tools when needed.", tools=[tool])
    print(await agent.run("What's the weather in Bengaluru?"))
    print(await agent.run("And Paris?"))

asyncio.run(main())
