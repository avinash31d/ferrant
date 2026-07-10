"""Durable fan-out, retry, join, approval, resume, and recovery."""
import asyncio
from liteagent import WorkflowBuilder

def plan(ctx: dict) -> dict:
    return {"update": {"plan": "review in parallel"}, "routes": ["research", "risk"]}

def research(ctx: dict) -> dict:
    return {"update": {"research": {"finding": "staged rollout reduces blast radius"}}}

def risk(ctx: dict) -> dict:
    if ctx["attempt"] == 1:
        raise RuntimeError("simulated transient dependency failure")
    return {"update": {"risk": {"finding": "retain rollback capacity"}}}

def publish(ctx: dict) -> dict:
    return {"update": {"published": True}}

def workflow():
    builder = WorkflowBuilder("release-workflow", ".liteagent/workflows", version="1")
    builder.entry("plan")
    builder.node("plan", plan)
    builder.node("research", research)
    builder.node("risk", risk, max_attempts=3, timeout_seconds=5.0)
    builder.node("publish", publish)
    builder.route("plan", "research", "research")
    builder.route("plan", "risk", "risk")
    builder.join(["research", "risk"], "publish")
    builder.interrupt_before("publish")
    return builder.build()

async def main() -> None:
    graph = workflow()
    checkpoint = await graph.run("release-42", {"release": "v2"})
    print("approval checkpoint:", checkpoint["status"], checkpoint["state"])
    checkpoint = await graph.resume("release-42", {"approved_by": "operator"})
    print("completed:", checkpoint["status"], checkpoint["state"])
    print("recovered:", (await graph.recover("release-42"))["status"])

asyncio.run(main())
