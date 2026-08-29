# Single image serving both the API and the Celery worker. They share the same
# code and dependencies and differ only in their command, so building one image
# halves build time and guarantees the two can never drift apart.
#
# Deliberately excludes scripts/requirements-ingest.txt. Playwright and its
# browser bundle add roughly 400 MB and ingestion runs offline from a developer
# machine, never inside the deployed service.
FROM python:3.13-slim

# PyMuPDF needs libgl for its rasteriser, and curl backs the container health
# check. --no-install-recommends keeps the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so dependency layers cache independently of source edits.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Run unprivileged. The callback dead-letter file is the only path written at
# runtime, so it gets its own directory owned by the app user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
