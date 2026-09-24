# docker build -t nl2sql-bank .
# docker run -p 8000:8000 -v nl2sql-data:/app/data --env-file .env nl2sql-bank
# The 200 MB warehouse is not baked in: the entrypoint builds it into the volume on first start.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NL2SQL_HOME=/app \
    NL2SQL_LOG_FORMAT=json

# curl is only for the HEALTHCHECK. The app runs unprivileged: it executes generated SQL.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 nl2sql

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[embeddings,ui]"

COPY app/ app/
COPY evaluation/ evaluation/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN mkdir -p data \
 && chown -R nl2sql:nl2sql data evaluation \
 && chmod +x /usr/local/bin/entrypoint.sh

USER nl2sql
VOLUME ["/app/data"]
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
