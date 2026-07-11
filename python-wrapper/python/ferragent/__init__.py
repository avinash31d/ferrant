"""Thin optional Python API backed by the native Rust ferragent runtime."""

from importlib.metadata import version

from ._native import Agent, McpTools, Retriever, Team, Tool, WorkflowBuilder, WorkflowGraph

__version__ = version("ferragent")

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
