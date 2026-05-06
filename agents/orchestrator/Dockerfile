FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# shared models mounted at runtime via Docker volume — not copied here
COPY . .

ENV PYTHONPATH="/shared:/app"

EXPOSE 8001
CMD ["uvicorn", "a2a_server:app", "--host", "0.0.0.0", "--port", "8001"]
