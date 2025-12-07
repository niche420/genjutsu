FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4
RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Shap-E
RUN git clone https://github.com/openai/shap-e.git && \
    cd shap-e && \
    pip install -e . && \
    cd .. && \
    rm -rf shap-e/.git

# Copy worker
COPY worker.py .

# Create output directory
RUN mkdir -p /app/outputs

CMD ["python3", "worker.py"]