# User Guide - AI Book Agent MCP Server

This guide covers setup, usage, and deployment of the AI Book Agent using the official Python MCP SDK with local models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Model Setup](#local-model-setup)
3. [MCP Server Configuration](#mcp-server-configuration)
4. [Using the Server](#using-the-server)
5. [Remote Access Setup](#remote-access-setup)
6. [Adding Books](#adding-books)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for large books)
- **Storage**: 10GB free disk space
- **Internet**: Required for initial model downloads

### Installation

#### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd ai-book-agent

# Run automated setup with UV
./setup-uv.sh
```

This script will:
- Install UV package manager
- Create Python 3.11 virtual environment
- Install all dependencies
- Set up Ollama and download Qwen model
- Create necessary directories

#### Option 2: Manual Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-book-agent

# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip sync requirements.txt

# Install Ollama for local models
curl -fsSL https://ollama.ai/install.sh | sh  # macOS/Linux
# Or visit https://ollama.ai for Windows installer

# Pull required models
ollama pull qwen2.5:7b
```

## Local Model Setup

### 1. Configuration

The system uses a single `config.yaml` file for all settings:

```yaml
# config.yaml
embeddings:
  model: "all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" if GPU available
  batch_size: 32
  cache_dir: "./models/embeddings"

generation:
  provider: "ollama"
  model: "qwen2.5:7b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 2048

books:
  data_dir: "data/epub"
  processed_dir: "data/processed"
  index_dir: "data/vector_db"

vector_store:
  provider: "chromadb"
  collection_name: "ml_books"
  chunk_size: 512
  chunk_overlap: 50

search:
  max_results: 5
  similarity_threshold: 0.5
  rerank: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 2. Verify Models

```bash
# Activate virtual environment
source .venv/bin/activate

# Test embeddings
python scripts/test_components.py

# Test search functionality
python scripts/test_search.py

# Test without Ollama (embeddings only)
python scripts/test_without_ollama.py
```

## MCP Server Configuration

### 1. Start the MCP Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Development mode with auto-reload
mcp dev server.py

# Production mode
python server.py
```

### 2. Configure Claude Desktop

Add this to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ai-book-agent": {
      "command": "/path/to/ai-book-agent/.venv/bin/python",
      "args": ["/path/to/ai-book-agent/server.py"]
    }
  }
}
```

**Note**: Update the paths to match your actual installation directory.

### 3. Verify MCP Connection

```bash
# Test MCP server is running
curl http://localhost:8080/health

# Test using MCP CLI
mcp dev server.py
# In another terminal:
mcp call ai-book-agent searchBooks --query "machine learning"
```

## Using the Server

### In Claude Desktop

Once configured, you can ask questions about ML topics and Claude will automatically use your book knowledge:

```
You: What are the best practices for ML model monitoring according to the textbooks?

Claude: I'll search your ML textbooks for monitoring best practices.
[Uses searchBooks tool automatically]

Based on your textbooks, here are the key monitoring practices...
```

### Available MCP Tools

The server provides these tools to Claude:

1. **searchBooks**: Search across all indexed books
2. **getChapterContent**: Retrieve specific book sections  
3. **generateSection**: Create documentation from sources
4. **citeSources**: Generate proper citations

### Command Line Testing

```bash
# Test search functionality
mcp call ai-book-agent searchBooks \
  --query "deployment strategies" \
  --maxResults 3

# Test chapter retrieval
mcp call ai-book-agent getChapterContent \
  --bookId "designing-ml-systems" \
  --chapterTitle "Deployment"

# Test documentation generation
mcp call ai-book-agent generateSection \
  --topic "ML Pipeline Best Practices" \
  --style "technical"
```

## Remote Access Setup

### Option 1: Cloudflare Tunnel (Recommended)

```bash
# Install cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Create tunnel
cloudflared tunnel create book-agent

# Configure tunnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: book-agent
credentials-file: /home/$USER/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: book-agent.yourdomain.com
    service: http://localhost:8080
  - service: http_status:404
EOF

# Run tunnel
cloudflared tunnel run book-agent
```

### Option 2: Nginx Reverse Proxy

```nginx
server {
    server_name book-agent.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-API-Key $http_x_api_key;
    }
    
    ssl_certificate /etc/letsencrypt/live/book-agent.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/book-agent.yourdomain.com/privkey.pem;
}
```

### Configure Remote Claude Desktop

```json
{
  "mcpServers": {
    "ai-book-agent-remote": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-proxy",
        "wss://book-agent.yourdomain.com/mcp"
      ],
      "env": {
        "BOOK_AGENT_API_KEY": "your-secure-key-here"
      }
    }
  }
}
```

## Adding Books

### 1. Add EPUB Files

```bash
# Activate virtual environment
source .venv/bin/activate

# Copy book to data directory
cp "Machine Learning Book.epub" data/epub/

# Index all books (processes new ones automatically)
python scripts/index_books.py
```

### 2. Verify Indexing

```bash
# Test search after adding books
python scripts/test_search.py

# Check what books are indexed
ls data/processed/

# Verify vector database
ls data/vector_db/
```

### 3. Supported Book Formats

- **EPUB**: Fully supported (recommended)
- **PDF**: Not yet supported
- **HTML**: Not yet supported

### 4. Book Processing

The system automatically:
- Extracts text from EPUB files
- Splits content into semantic chunks
- Generates embeddings using sentence-transformers
- Stores in ChromaDB vector database
- Maintains metadata for citations

## Advanced Usage

### Custom Prompts

```yaml
# config/prompts/custom.yaml
prompts:
  architecture_doc:
    name: "ML Architecture Document"
    template: |
      Create an architecture document for {system_name} that includes:
      1. System overview based on ML best practices
      2. Component design from textbook patterns
      3. Deployment architecture
      
    parameters:
      - system_name
      - requirements
```

### Batch Processing

```python
# scripts/batch_generate.py
topics = [
    "Data Pipeline Design",
    "Feature Store Architecture", 
    "Model Registry Setup"
]

for topic in topics:
    response = agent.generate_section(
        topic=topic,
        output_dir="./docs/generated"
    )
```

### Performance Tuning

```yaml
# config/performance.yaml
cache:
  enabled: true
  ttl_seconds: 3600
  max_size_mb: 1000

chunking:
  size: 512  # Smaller chunks for faster search
  overlap: 50

search:
  rerank: true
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
```bash
# Check Ollama is running
sudo systemctl status ollama

# Restart if needed
sudo systemctl restart ollama
```

2. **Memory Issues**
```bash
# Reduce model size
ollama pull qwen2.5:3b  # Smaller model

# Or limit memory
export OLLAMA_MAX_MEMORY=8G
```

3. **Slow Embeddings**
```python
# Use GPU if available
# In config/embeddings.yaml
device: "cuda"  # Instead of "cpu"
```

4. **MCP Connection Issues**
```bash
# Test MCP server
curl http://localhost:8080/health

# Check logs
tail -f logs/mcp-server.log
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
npm start

# Test individual components
python -m src.test_component --component embeddings
python -m src.test_component --component search
```

### Performance Monitoring

```bash
# Monitor resource usage
htop

# Check model memory
nvidia-smi  # If using GPU

# Monitor API calls
tail -f logs/api-access.log | grep -E "searchBooks|generateSection"
```

## Best Practices

1. **Security**
   - Always use strong API keys
   - Enable HTTPS for remote access
   - Regularly update dependencies

2. **Performance**
   - Pre-compute embeddings for new books
   - Use caching for frequent queries
   - Monitor resource usage

3. **Maintenance**
   - Backup vector index regularly
   - Keep audit logs
   - Test after adding new books

## Support

- GitHub Issues: [Report bugs and feature requests]
- Documentation: Check `/docs` folder
- Logs: See `/logs` directory for debugging