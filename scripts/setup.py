#!/usr/bin/env python3
"""Setup script for AI Book Agent."""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Main setup function."""
    logger.info("Setting up AI Book Agent...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        logger.error("Python 3.11+ is required")
        sys.exit(1)
    
    logger.info(f"Python version: {sys.version}")
    
    # Install dependencies
    logger.info("Installing dependencies...")
    
    deps = [
        'pip install --upgrade pip',
        'pip install "mcp[cli]"',
        'pip install sentence-transformers',
        'pip install chromadb',
        'pip install ollama',
        'pip install ebooklib',
        'pip install beautifulsoup4',
        'pip install pyyaml',
        'pip install numpy',
        'pip install pandas',
        'pip install pytest',
        'pip install pytest-asyncio'
    ]
    
    for dep in deps:
        if not run_command(dep, f"Installing {dep.split()[-1]}"):
            logger.error("Failed to install dependencies")
            sys.exit(1)
    
    # Check Ollama installation
    logger.info("Checking Ollama installation...")
    
    ollama_check = subprocess.run("which ollama", shell=True, capture_output=True)
    if ollama_check.returncode != 0:
        logger.warning("Ollama not found. Please install it manually:")
        logger.warning("curl -fsSL https://ollama.ai/install.sh | sh")
        logger.warning("Then run: ollama pull qwen2.5:7b")
    else:
        logger.info("✅ Ollama found")
        
        # Check if Qwen model is available
        qwen_check = subprocess.run("ollama list | grep qwen2.5:7b", shell=True, capture_output=True)
        if qwen_check.returncode != 0:
            logger.info("Pulling Qwen 2.5 7B model...")
            if not run_command("ollama pull qwen2.5:7b", "Pulling Qwen model"):
                logger.warning("Failed to pull Qwen model. You can do this manually later.")
        else:
            logger.info("✅ Qwen 2.5 7B model found")
    
    # Create directories
    directories = [
        "data/epub",
        "data/processed", 
        "data/vector_db",
        "models/embeddings",
        "tests",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    # Test basic imports
    logger.info("Testing imports...")
    
    try:
        import mcp
        import sentence_transformers
        import chromadb
        import ollama
        import ebooklib
        import bs4
        import yaml
        logger.info("✅ All imports successful")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        sys.exit(1)
    
    logger.info("\n🎉 Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Place EPUB files in data/epub/")
    logger.info("2. Run: python scripts/index_books.py")
    logger.info("3. Test: python scripts/test_components.py") 
    logger.info("4. Start server: python server.py")
    logger.info("\nFor Claude Desktop integration, add this to your config:")
    logger.info('{')
    logger.info('  "mcpServers": {')
    logger.info('    "ai-book-agent": {')
    logger.info('      "command": "python",')
    logger.info(f'      "args": ["{Path.cwd() / "server.py"}"]')
    logger.info('    }')
    logger.info('  }')
    logger.info('}')

if __name__ == "__main__":
    main()