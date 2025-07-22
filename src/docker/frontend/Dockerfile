FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/frontend/requirements.txt /app/frontend/

# Install dependencies using correct path
RUN pip install --upgrade pip && \
    pip install -r /app/frontend/requirements.txt

# Copy rest of the application
COPY src/frontend /app/frontend/

EXPOSE 8080

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8080"]
