FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/MCP_client/requirements.txt /app/MCP_client/

# Install dependencies using correct path
RUN pip install --upgrade pip && \
    pip install -r /app/MCP_client/requirements.txt

# Copy rest of the application
COPY src/MCP_client /app/MCP_client/

EXPOSE 8002

CMD ["python", "MCP_client/client.py"]
