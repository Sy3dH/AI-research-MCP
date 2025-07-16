FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/MCP_server/requirements.txt /app/MCP_server/

# Install dependencies using correct path
RUN pip install --upgrade pip && \
    pip install -r /app/MCP_server/requirements.txt

# Copy rest of the application
COPY src/MCP_server /app/MCP_server/

# Pre-download FastEmbed model
RUN python3 -c "\
from fastembed import TextEmbedding; \
embedder = TextEmbedding(model_name='BAAI/bge-base-en-v1.5'); \
embedder.embed(['Pre-warm model download'])"

EXPOSE 8080

CMD ["python", "MCP_server/server.py"]
