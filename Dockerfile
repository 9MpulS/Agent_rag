FROM python:3.12-slim

# Install system dependencies for PDF parsing and OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ukr \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies without the project itself to cache this layer
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project
RUN uv sync --frozen

# Expose FastAPI port
EXPOSE 8000

# Run the application using uv run
CMD ["uv", "run", "python", "main.py"]
