FROM python:3.10-slim

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libmupdf-dev \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements to leverage Docker cache for dependencies
COPY requirements.txt ./

# Install dependencies: CPU-only torch first, then others
RUN pip install --no-cache-dir torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt')"

# Copy model and application code
COPY models/Gemma-1B.Q4_K_M.gguf /models/Gemma-1B.Q4_K_M.gguf
COPY . /app

# Entrypoint
ENTRYPOINT ["python", "main.py"]
