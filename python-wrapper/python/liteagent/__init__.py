"""Thin optional Python API backed by the native Rust liteagent runtime."""

from importlib.metadata import version

from ._native import Agent, McpTools, Retriever, Team, Tool, WorkflowBuilder, WorkflowGraph

__version__ = version("liteagent")

__all__ = [
    "Agent",
    "McpTools",
    "Retriever",
    "Team",
    "Tool",
    "WorkflowBuilder",
    "WorkflowGraph",
    "__version__",
]
