import tempfile
import unittest
from pathlib import Path

from ferrant import Retriever, Tool, WorkflowBuilder


class WrapperSmokeTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_construction(self) -> None:
        tool = Tool("echo", "Echo JSON", {"type": "object"}, lambda value: value)
        self.assertEqual(tool.name, "echo")

    async def test_retrieval_runs_in_rust(self) -> None:
        retriever = Retriever()
        await retriever.upsert([
            {"id": "durability", "text": "checkpoints enable durable recovery", "metadata": {}}
        ])
        results = await retriever.retrieve("durable recovery", limit=1)
        self.assertEqual(results[0]["document"]["id"], "durability")

    async def test_graph_pause_resume_and_recover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = WorkflowBuilder("smoke", str(Path(directory) / "graphs"))
            builder.entry("prepare")
            builder.node("prepare", lambda _ctx: {"update": {"prepared": True}})
            builder.node("publish", lambda _ctx: {"update": {"published": True}})
            builder.edge("prepare", "publish")
            builder.interrupt_before("publish")
            graph = builder.build()

            paused = await graph.run("run-1", {})
            self.assertEqual(paused["status"], "paused")
            completed = await graph.resume("run-1", {"approved": True})
            self.assertEqual(completed["status"], "completed")
            self.assertTrue((await graph.recover("run-1"))["state"]["published"])


if __name__ == "__main__":
    unittest.main()
