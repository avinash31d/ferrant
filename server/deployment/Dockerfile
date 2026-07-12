FROM python:3.12-slim

WORKDIR /srv/ferrant
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deployment/app.py ./app.py
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
