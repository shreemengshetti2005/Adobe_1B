FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt ./

# Copy the Gemma model (GGUF)
COPY models/Gemma-1B.Q4_K_M.gguf /models/Gemma-1B.Q4_K_M.gguf

# Install Torch (CPU only)
RUN pip install --no-cache-dir torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (punkt tokenizer)
RUN python -m nltk.downloader punkt

# (Option 1) Pre-download SentenceTransformer model during image build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of your project
COPY . /app

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
