"""Thin optional Python API backed by the native Rust ferrant runtime."""

from importlib.metadata import version

from ._native import Agent, McpTools, Retriever, Team, Tool, WorkflowBuilder, WorkflowGraph

__version__ = version("ferrant")

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
