"""Ferrant's deployment API. This service is the only component that talks to Docker."""
import io
import os
import re
import tarfile
import time
import uuid
from pathlib import Path

import docker
from fastapi import FastAPI, Header, HTTPException, Request


app = FastAPI(title="Ferrant deployment server")
client = docker.from_env()
RUNNER_IMAGE = os.getenv("FERRANT_RUNNER_IMAGE", "ferrant-runner:latest")
PUBLIC_HOST = os.getenv("FERRANT_PUBLIC_HOST")
MAX_PACKAGE_BYTES = int(os.getenv("FERRANT_MAX_PACKAGE_BYTES", str(50 * 1024 * 1024)))
# This must be a host path also mounted at the identical path in this API
# container; Docker bind mounts are resolved by the Docker daemon, not here.
APP_ROOT = Path(os.getenv("FERRANT_APP_ROOT", "/srv/ferrant/apps"))


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:40] or "app"


def unpack(source: bytes, destination: Path) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                    raise ValueError(f"unsafe archive path: {member.name}")
            archive.extractall(destination, filter="data")
    except (tarfile.TarError, ValueError) as error:
        raise HTTPException(400, f"invalid application package: {error}") from error


def wait_for_health(container) -> None:
    for _ in range(30):
        container.reload()
        if container.status == "running":
            return
        time.sleep(0.5)
    raise RuntimeError("runner container did not start")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/deployments", status_code=201)
async def create_deployment(
    request: Request,
    x_ferrant_name: str = Header(...),
    x_ferrant_handler: str = Header(...),
):
    if ":" not in x_ferrant_handler:
        raise HTTPException(400, "handler must use module:function syntax")
    source = await request.body()
    if not source or len(source) > MAX_PACKAGE_BYTES:
        raise HTTPException(413, "application package is empty or too large")
    if not PUBLIC_HOST:
        raise HTTPException(500, "FERRANT_PUBLIC_HOST must be configured on the deployment server")

    deployment_id = uuid.uuid4().hex[:12]
    name = f"ferrant-{safe_name(x_ferrant_name)}-{deployment_id}"
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = APP_ROOT / deployment_id
    workspace.mkdir()
    unpack(source, workspace)
    try:
        container = client.containers.run(
            RUNNER_IMAGE,
            name=name,
            detach=True,
            remove=True,
            labels={"ferrant.managed": "true", "ferrant.app": x_ferrant_name},
            ports={"8000/tcp": None},
            volumes={str(workspace): {"bind": "/app", "mode": "ro"}},
            working_dir="/app",
            command=["python", "/opt/ferrant/ferrant_server.py", "--app-dir", "/app", "--handler", x_ferrant_handler, "--port", "8000"],
        )
        wait_for_health(container)
        container.reload()
        port = container.attrs["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"]
    except Exception as error:
        try:
            container.remove(force=True)
        except (UnboundLocalError, docker.errors.DockerException):
            pass
        raise HTTPException(500, f"could not start deployment: {error}") from error
    return {"id": deployment_id, "endpoint": f"{PUBLIC_HOST.rstrip('/')}:{port}"}
