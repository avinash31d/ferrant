FROM python:3.12-slim

WORKDIR /app
COPY runtime/ferrant_server.py /opt/ferrant/ferrant_server.py
RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000
CMD ["sleep", "infinity"]
