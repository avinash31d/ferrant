"""Console-script entry point backed by Ferrant's native Rust CLI."""
import sys

from ._native import run_cli


def main() -> None:
    raise SystemExit(run_cli(sys.argv[1:]))
