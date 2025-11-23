#!/bin/bash
# Quick Docker setup for GaussianDreamer service

set -e

echo "╔════════════════════════════════════════╗"
echo "║  GaussianDreamer Docker Setup          ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✓ Docker found: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found"
    exit 1
fi

echo "✓ Docker Compose found"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU (slower)"
fi

echo ""
echo "Building Docker image (this may take 10-15 minutes)..."
echo ""

# Build
docker-compose build

echo ""
echo "✓ Build complete!"
echo ""
echo "Starting service..."

# Start
docker-compose up -d

echo ""
echo "Waiting for service to be ready..."

# Check health
for i in {1..10}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo ""
        echo "╔════════════════════════════════════════╗"
        echo "║  ✓ Service Ready!                      ║"
        echo "╚════════════════════════════════════════╝"
        echo ""
        echo "Service running at: http://localhost:5000"
        echo ""
        echo "Test it:"
        echo "  curl http://localhost:5000/health"
        echo ""
        echo "View logs:"
        echo "  docker-compose logs -f"
        echo ""
        echo "Stop service:"
        echo "  docker-compose down"
        echo ""
        echo "Now run your Rust app:"
        echo "  cargo run --release"
        echo ""
        exit 0
    fi
    echo "  Waiting... ($i/10)"
    sleep 3
done

echo ""
echo "⚠️  Service started but not responding yet"
echo "Check logs: docker-compose logs gaussiandreamer"
