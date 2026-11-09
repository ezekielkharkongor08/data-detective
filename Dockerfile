# Data Detective API image.
# Builds a single image used by both the FastAPI and Streamlit services.
# Local demo setup — not production hardened.

FROM python:3.13-slim

WORKDIR /app

# System dependency for psycopg binary (PostgreSQL driver).
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the project's dependency file, no cache.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API and UI need at runtime.
COPY src ./src
COPY main.py ./
COPY ui ./ui

# Create a non-root user and writable runtime directories.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/uploads /app/plots && \
    chown -R appuser:appuser /app/data /app/uploads /app/plots /app/src /app/main.py /app/ui

USER appuser

# Ollama runs on the host, not in this container.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
# Default API URL for Streamlit to connect to host-mapped FastAPI
ENV API_URL=http://host.docker.internal:8000
# Streamlit server defaults
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
# SQLite default: database file lives in the data volume.
ENV DATABASE_PATH=/app/data/data_detective.db
# DATABASE_URL is intentionally not set here: an unset/empty value selects SQLite.
# Set it in docker-compose or .env to use the Postgres service instead.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Default command runs the FastAPI API. The Streamlit service overrides this.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
