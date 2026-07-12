# Ferrant deployment server

`runtime/ferrant_server.py` is the generic FastAPI/Uvicorn inference server.
It imports the user's `handler: module:function` from `/app` and exposes
`GET /health` and `POST /infer`.

Build the reusable runner once:

```bash
docker build -t ferrant-runner:latest -f server/docker/Dockerfile server
```

The CLI never invokes Docker. It packages the application directory as a gzip
tarball and uploads it to this FastAPI deployment API. The deployment server
unpacks the package in a temporary directory and starts a runner container
using the host's Docker daemon.

Build the runner and the deployment service, then run the service with access
to the Docker socket. `FERRANT_PUBLIC_HOST` must be the scheme and hostname
reachable by clients (for example `https://agents.example.com`).

```bash
docker build -t ferrant-runner:latest -f server/docker/Dockerfile server
docker build -t ferrant-deployment-server:latest -f server/deployment/Dockerfile server
docker run -d --restart unless-stopped \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /srv/ferrant/apps:/srv/ferrant/apps \
  -e FERRANT_PUBLIC_HOST=https://agents.example.com \
  ferrant-deployment-server:latest
```

Deploy from a development machine with:

```bash
ferrant deploy --server https://deploy.example.com
```

The deployment API is privileged because it can create Docker containers. Put
it behind authentication and do not expose it directly to untrusted users.
