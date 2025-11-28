#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║        Local Setup - Multi-Model Service                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Create directories
mkdir -p repositories outputs models

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Clone GaussianDreamer
echo ""
echo "Cloning GaussianDreamer..."
if [ ! -d "repositories/GaussianDreamer" ]; then
    git clone https://github.com/hustvl/GaussianDreamer.git repositories/GaussianDreamer
    cd repositories/GaussianDreamer
    pip install -r requirements.txt
    cd ../..
else
    echo "  ✓ Already exists"
fi

# Clone DreamGaussian
echo ""
echo "Cloning DreamGaussian..."
if [ ! -d "repositories/dreamgaussian" ]; then
    git clone https://github.com/dreamgaussian/dreamgaussian.git repositories/dreamgaussian
    cd repositories/dreamgaussian
    pip install -r requirements.txt
    cd ../..
else
    echo "  ✓ Already exists"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Run the service:"
echo "  python multi_model_service.py"
echo ""
echo "Test it:"
echo "  curl http://localhost:5000/health"
echo ""