FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/shared:/app"

# Port 8003 = synthesis A2A server
# Ports 8011-8013 = MCP tool servers (started separately in docker-compose)
EXPOSE 8003
CMD ["uvicorn", "a2a_server:app", "--host", "0.0.0.0", "--port", "8003"]
