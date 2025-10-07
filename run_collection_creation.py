#!/usr/bin/env python3
"""
Docker-safe Collection Creation Runner
Handles ChromaDB collection creation with proper Docker environment considerations
"""
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_docker_running():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH")
        return False

def check_containers_running():
    """Check if any AUB-RAG-BE containers are running"""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=aub-rag-be', '-q'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def stop_containers():
    """Stop running containers"""
    try:
        logger.info("Stopping existing containers...")
        result = subprocess.run(['docker', 'compose', 'down'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Containers stopped successfully")
            return True
        else:
            logger.error(f"Failed to stop containers: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error stopping containers: {e}")
        return False

def build_image_if_needed():
    """Build Docker image if it doesn't exist"""
    try:
        # Check if image exists
        result = subprocess.run(['docker', 'images', '-q', 'aub-rag-be-web'], 
                              capture_output=True, text=True)
        if not result.stdout.strip():
            logger.info("Building Docker image...")
            result = subprocess.run(['docker', 'compose', 'build', 'web'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to build image: {result.stderr}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error building image: {e}")
        return False

def run_collection_creation():
    """Run the collection creation script in Docker"""
    try:
        current_dir = Path.cwd()
        logger.info("Running collection creation script...")
        
        cmd = [
            'docker', 'run', '--rm',
            '-v', f"{current_dir}/chroma_db:/app/chroma_db",
            '-v', f"{current_dir}/sonu_resources:/app/sonu_resources", 
            '-v', f"{current_dir}/.env:/app/.env",
            '--env-file', '.env',
            '-e', 'PYTHONUNBUFFERED=1',
            'aub-rag-be-web',
            'python3', 'create_sonu_qa_v3_docker_safe.py'
        ]
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running collection creation: {e}")
        return False

def restart_containers():
    """Restart the containers"""
    try:
        logger.info("Restarting containers...")
        result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Containers restarted successfully")
            return True
        else:
            logger.error(f"Failed to restart containers: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error restarting containers: {e}")
        return False

def main():
    """Main execution function"""
    logger.info("🚀 Starting SONU QA V3 Collection Creation Process...")
    
    # Check Docker
    if not check_docker_running():
        logger.error("❌ Docker is not running. Please start Docker first.")
        sys.exit(1)
    
    # Build image if needed
    if not build_image_if_needed():
        logger.error("❌ Failed to build Docker image")
        sys.exit(1)
    
    # Check if containers are running
    containers_were_running = check_containers_running()
    
    if containers_were_running:
        logger.info("📦 Containers are running, stopping them first...")
        if not stop_containers():
            logger.error("❌ Failed to stop containers")
            sys.exit(1)
        # Wait for resources to be released
        time.sleep(3)
    else:
        logger.info("ℹ️  No containers running")
    
    # Run collection creation
    if run_collection_creation():
        logger.info("✅ Collection creation completed successfully!")
    else:
        logger.error("❌ Collection creation failed!")
        sys.exit(1)
    
    # Restart containers if they were running
    if containers_were_running:
        logger.info("🔄 Restarting containers...")
        if not restart_containers():
            logger.warning("⚠️  Collection created but failed to restart containers")
            logger.info("💡 You can manually restart with: docker compose up -d")
            sys.exit(1)
    else:
        logger.info("ℹ️  Containers were not running, leaving them stopped")
        logger.info("💡 To start your application: docker compose up -d")
    
    logger.info("🎉 SONU QA V3 Collection creation process completed!")

if __name__ == "__main__":
    main()
