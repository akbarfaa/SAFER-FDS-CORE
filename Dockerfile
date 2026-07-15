FROM python:3.10-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Set environment variables for production/sandbox mode
ENV PORT=7860
ENV RENDER=true
ENV IS_PRODUCTION=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Hugging Face Spaces expects application on port 7860
EXPOSE 7860

# Launch services
CMD ["python", "start_services.py"]
