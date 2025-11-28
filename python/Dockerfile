FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app

# Clone model repositories
RUN git clone https://github.com/hustvl/GaussianDreamer.git repositories/GaussianDreamer && \
    cd repositories/GaussianDreamer && \
    pip install -r requirements.txt && \
    cd ../..

RUN git clone https://github.com/dreamgaussian/dreamgaussian.git repositories/dreamgaussian && \
    cd repositories/dreamgaussian && \
    pip install -r requirements.txt && \
    cd ../..

# Copy service files
COPY requirements.txt .
COPY models/ models/
COPY multi_model_service.py .

# Install service dependencies
RUN pip install -r requirements.txt

# Create output directory
RUN mkdir -p outputs

EXPOSE 5000

CMD ["python3", "multi_model_service.py", "--host", "0.0.0.0", "--port", "5000"]