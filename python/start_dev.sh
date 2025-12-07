#!/bin/bash
# Quick development startup script

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           Genjutsu - Development Startup                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd python

# Check if services are already running
if docker-compose ps | grep -q "Up"; then
    echo "⚠️  Docker services already running"
    read -p "Do you want to restart them? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing services..."
        docker-compose down
    else
        echo "Keeping existing services"
        exit 0
    fi
fi

# Start services
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 5

# Check health
echo ""
echo "Checking service health..."
if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
    echo "✓ FastAPI: http://localhost:5000"
    echo "✓ API Docs: http://localhost:5000/docs"
else
    echo "✗ FastAPI health check failed"
    echo "  Run 'docker-compose logs api' to check logs"
fi

echo ""
echo "Checking workers..."
WORKERS=$(curl -s http://localhost:5000/workers | grep -o '"workers":.*' | grep -o '{' | wc -l)
if [ "$WORKERS" -gt 0 ]; then
    echo "✓ Celery workers: $WORKERS active"
else
    echo "✗ No workers found"
    echo "  Run 'docker-compose logs worker' to check logs"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Services started!                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. View logs: docker-compose logs -f"
echo "  2. Test API: curl http://localhost:5000/health"
echo "  3. Run Rust app: cargo run --release"
echo ""
echo "To stop services: docker-compose down"
echo ""