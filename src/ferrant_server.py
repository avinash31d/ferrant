"""Python inference server embedded in the Ferrant CLI."""

import argparse
import importlib
import inspect
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import RootModel
import uvicorn


def load_handler(app_dir: str, reference: str):
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("handler must be written as module:function")
    sys.path.insert(0, str(Path(app_dir).resolve()))
    handler = getattr(importlib.import_module(module_name), function_name)
    if not callable(handler):
        raise TypeError(f"{reference} is not callable")
    return handler


def make_app(handler):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/infer")
    async def infer(input_value: RootModel[dict]):
        try:
            output = handler(input_value.root)
            if inspect.isawaitable(output):
                output = await output
            if not isinstance(output, dict):
                raise TypeError("agent handler must return a dict")
            return output
        except Exception as error:
            raise HTTPException(400, str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", required=True)
    parser.add_argument("--handler", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = make_app(load_handler(args.app_dir, args.handler))
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
