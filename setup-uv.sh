#!/bin/bash
set -e

echo "🚀 Setting up AI Book Agent with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add UV to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ UV is installed"

# Create virtual environment with Python 3.11
echo "🐍 Creating virtual environment with Python 3.11..."
uv venv --python 3.11

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Sync dependencies from pyproject.toml
echo "📦 Installing dependencies..."
uv pip sync requirements.txt

# Install development dependencies
echo "🛠️  Installing development dependencies..."
uv pip install pytest pytest-asyncio ipykernel notebook black ruff mypy

# Check Ollama installation
echo "🤖 Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install it manually:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    echo "   Then run: ollama pull qwen2.5:7b"
else
    echo "✅ Ollama found"
    # Check if Qwen model is available
    if ! ollama list | grep -q "qwen2.5:7b"; then
        echo "📥 Pulling Qwen 2.5 7B model..."
        ollama pull qwen2.5:7b
    else
        echo "✅ Qwen 2.5 7B model found"
    fi
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/epub data/processed data/vector_db models/embeddings tests logs

echo "
🎉 Setup completed successfully!

To activate the virtual environment in future sessions:
    source .venv/bin/activate

Next steps:
1. Place EPUB files in data/epub/
2. Run: python scripts/index_books.py
3. Test: python scripts/test_components.py
4. Start server: python server.py

For Claude Desktop integration, add this to your config:
{
  \"mcpServers\": {
    \"ai-book-agent\": {
      \"command\": \"$PWD/.venv/bin/python\",
      \"args\": [\"$PWD/server.py\"]
    }
  }
}
"