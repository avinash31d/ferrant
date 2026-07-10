import asyncio
import os
from liteagent import Agent, Tool

def calculate(args: dict) -> str:
    a, b, op = float(args["a"]), float(args["b"]), args["op"]
    operations = {"+": lambda: a + b, "-": lambda: a - b,
                  "*": lambda: a * b, "/": lambda: a / b}
    return str(operations[op]())

async def main() -> None:
    calculator = Tool("calculator", "Calculate with two numbers", {
        "type": "object", "properties": {"a": {"type": "number"},
        "op": {"enum": ["+", "-", "*", "/"]}, "b": {"type": "number"}},
        "required": ["a", "op", "b"]}, calculate)
    agent = Agent.openai("gpt-5-nano", os.environ["OPENAI_API_KEY"],
        instructions="Always use the calculator for arithmetic.", tools=[calculator],
        storage_path=".liteagent/sessions")
    print(await agent.run_session("user-42", "What is 42 * 17?"))
    print(await agent.run_session("user-42", "Now subtract 100."))

asyncio.run(main())
