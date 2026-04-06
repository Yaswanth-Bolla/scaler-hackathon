FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic websockets openai requests

# Copy environment code
COPY . /app/incident_env/

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "incident_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
