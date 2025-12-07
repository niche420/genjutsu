#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║        Genjutsu - Local Setup (Redis + Celery)          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "❌ Redis not found!"
    echo ""
    echo "Please install Redis:"
    echo "  macOS:   brew install redis"
    echo "  Ubuntu:  sudo apt install redis-server"
    echo "  Windows: Use WSL2 or Docker"
    exit 1
fi

echo "✓ Redis found: $(redis-server --version | head -n1)"
echo ""

# Create conda environment
echo "Creating conda environment 'genjutsu'..."
if conda env list | grep -q "^genjutsu "; then
    echo "Environment 'genjutsu' already exists."
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n genjutsu -y
    else
        echo "Keeping existing environment."
        exit 0
    fi
fi

# Create environment
echo "Creating environment with Python 3.12..."
conda create -n genjutsu python=3.12 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate genjutsu

# Install PyTorch with CUDA 12.4
echo ""
echo "Installing PyTorch with CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install API dependencies
echo ""
echo "Installing FastAPI dependencies..."
pip install -r api/requirements.txt

# Install Worker dependencies
echo ""
echo "Installing Worker dependencies..."
pip install -r worker/requirements.txt

# Install Shap-E
echo ""
echo "Installing Shap-E..."
if [ -d "shap-e" ]; then
    echo "Shap-E directory already exists, skipping clone..."
else
    git clone https://github.com/openai/shap-e.git
fi
cd shap-e
pip install -e .
cd ..

# Create outputs directory
mkdir -p ../outputs

# Create local Redis config
if [ ! -f "redis/redis.conf" ]; then
    echo ""
    echo "Creating Redis config..."
    mkdir -p redis
    cp redis/redis.conf.example redis/redis.conf 2>/dev/null || true
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "To run the services locally:"
echo ""
echo "Terminal 1 - Start Redis:"
echo "  redis-server python/redis/redis.conf"
echo ""
echo "Terminal 2 - Start Celery Worker:"
echo "  conda activate genjutsu"
echo "  cd python/worker"
echo "  python worker.py"
echo ""
echo "Terminal 3 - Start FastAPI:"
echo "  conda activate genjutsu"
echo "  cd python/api"
echo "  uvicorn main:app --host 0.0.0.0 --port 5000 --reload"
echo ""
echo "Terminal 4 - Run Rust app:"
echo "  cargo run --release"
echo ""
echo "Test the API:"
echo "  curl http://localhost:5000/health"
echo ""