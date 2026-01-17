# Engram API Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e ".[dev]"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["python", "-m", "engram.api.main"]
