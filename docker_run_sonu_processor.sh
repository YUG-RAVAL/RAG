#!/bin/bash
# Docker Script to Run SONU PDF Processor
# This script runs the SONU PDF processor inside the Docker container

echo "🚀 Running SONU PDF Processor in Docker Container..."

# Find the running web container
CONTAINER_NAME="aub-rag-be-web-1"

# Check if the specific container is running
if docker ps --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
    echo "✅ Found running container: $CONTAINER_NAME"
    echo "🔄 Executing SONU PDF processor..."
    
    # Run the processor directly in the container
    docker exec $CONTAINER_NAME python3 sonu_pdf_processor.py
    
    if [ $? -eq 0 ]; then
        echo "✅ SONU PDF processing completed successfully!"
    else
        echo "❌ SONU PDF processing failed!"
        exit 1
    fi
else
    echo "❌ Container '$CONTAINER_NAME' is not running."
    echo "Available containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    echo ""
    echo "Please make sure your container is running:"
    echo "  docker-compose up -d"
    exit 1
fi
