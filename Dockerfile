FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn

# Copy application
COPY organic_web_stress.py .

# Expose port
EXPOSE 7777

# Run application
CMD ["uvicorn", "organic_web_stress:app", "--host", "0.0.0.0", "--port", "7777", "--workers", "1"]