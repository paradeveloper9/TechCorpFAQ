# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "fastapi>=0.135.2" \
    "uvicorn[standard]>=0.42.0" \
    "pydantic>=2.12.5" \
    "pydantic-settings>=2.13.1" \
    "openai>=2.30.0" \
    "numpy>=2.4.4"

COPY src ./src
COPY data ./data

RUN useradd --create-home --no-log-init appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
