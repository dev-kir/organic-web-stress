FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy application
COPY organic_web_stress.py organic_web_stress.py

# Expose port
EXPOSE 7777

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7777/health')"

# Run with single worker (multiprocessing handles CPU cores internally)
CMD ["uvicorn", "organic_web_stress:app", "--host", "0.0.0.0", "--port", "7777"]