# User Guide - AI Book Agent

This guide provides comprehensive instructions for using the AI Book Agent system as a standalone MCP server.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Adding New Books](#adding-new-books)
4. [Using the MCP Server](#using-the-mcp-server)
5. [Available Tools](#available-tools)
6. [Extending the System](#extending-the-system)
7. [Troubleshooting](#troubleshooting)

## System Overview

The AI Book Agent is an MCP (Model Context Protocol) server that provides ML textbook knowledge through local models:

```
[EPUB Books] → [Parser] → [Chunker] → [Local Embeddings] → [Vector Store]
                                                                ↓
[Claude via MCP] → [Search Tools] → [Relevant Chunks] → [Local LLM] → [Response]
```

### Key Components

1. **MCP Server**: Exposes tools to Claude Desktop
2. **EPUB Parser**: Extracts structured content from books
3. **Sentence Transformers**: Creates local embeddings (no API calls)
4. **ChromaDB**: Vector database for semantic search
5. **Ollama**: Local LLM for text generation (Qwen 2.5 7B)
6. **MCP Tools**: searchBooks, getChapterContent, generateSection, citeSources

## Installation

### Prerequisites

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free disk space

### Quick Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-book-agent

# Run automated setup
./setup-uv.sh
```

This installs:
- UV package manager
- Python 3.11 virtual environment
- All dependencies (no external API keys needed)
- Ollama with Qwen 2.5 7B model
- Sentence transformers for embeddings

### Manual Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip sync requirements.txt

# Install and configure Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b

# Process existing books
python scripts/index_books.py
```

## Adding New Books

### Step 1: Add EPUB Files

Place EPUB files in the `data/epub/` directory:

```bash
cp "Machine Learning Book.epub" data/epub/
```

### Step 2: Index the Books

```bash
# Activate virtual environment
source .venv/bin/activate

# Process all books (detects new ones automatically)
python scripts/index_books.py
```

### Step 3: Verify Processing

```bash
# Test search functionality
python scripts/test_search.py

# Check processed files
ls data/processed/

# Verify vector database
ls data/vector_db/
```

### Supported Formats

- **EPUB**: Fully supported (recommended)
- **PDF**: Not yet supported
- **HTML**: Not yet supported

The system automatically:
- Extracts text and structure from EPUB files
- Splits content into semantic chunks
- Generates embeddings using local sentence-transformers
- Stores in ChromaDB vector database

## Using the MCP Server

### Start the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start MCP server
python server.py

# Or development mode with auto-reload
mcp dev server.py
```

### Configure Claude Desktop

Add to your Claude Desktop configuration:

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

### Using with Claude

Once configured, simply ask questions in Claude Desktop:

```
You: What are ML monitoring best practices according to the textbooks?

Claude: I'll search your ML textbooks for monitoring best practices.
[Uses searchBooks tool automatically]

Based on your textbooks, here are the key practices...
```

## Available Tools

The MCP server provides these tools to Claude:

### 1. searchBooks
Search across all indexed books for relevant content.

```bash
# Test via CLI
mcp call ai-book-agent searchBooks \
  --query "feature engineering" \
  --maxResults 5
```

### 2. getChapterContent
Retrieve specific sections from books.

```bash
mcp call ai-book-agent getChapterContent \
  --bookId "designing-ml-systems" \
  --chapterTitle "Deployment"
```

### 3. generateSection
Create documentation from multiple sources.

```bash
mcp call ai-book-agent generateSection \
  --topic "ML Pipeline Best Practices" \
  --style "technical"
```

### 4. citeSources
Generate proper citations for content.

## Extending the System

### Configuration

Modify `config.yaml` to customize behavior:

```yaml
# Use GPU for faster embeddings
embeddings:
  device: "cuda"  # Instead of "cpu"
  batch_size: 64  # Larger batches

# Adjust search parameters
search:
  max_results: 10
  similarity_threshold: 0.3  # Lower = more results

# Use different model
generation:
  model: "qwen2.5:14b"  # Larger model
```

### Adding New MCP Tools

Extend the server by adding tools in `server.py`:

```python
@mcp_server.tool()
async def custom_search_tool(query: str, book_filter: str = None) -> str:
    """Custom search with book filtering"""
    # Your implementation here
    pass
```

### Custom Processing Scripts

Create scripts in the `scripts/` directory:

```python
# scripts/batch_process.py
import asyncio
from src.search.vector_store import VectorStore

async def batch_query(queries: list):
    """Process multiple queries at once"""
    vector_store = VectorStore()
    results = []
    for query in queries:
        result = await vector_store.search(query)
        results.append(result)
    return results
```

## Troubleshooting

### Common Issues

1. **Virtual Environment Issues**
   ```bash
   # Recreate virtual environment
   rm -rf .venv
   ./setup-uv.sh
   ```

2. **Ollama Not Running**
   ```bash
   # Check Ollama status
   ollama list
   
   # Start Ollama service
   ollama serve
   
   # Pull missing models
   ollama pull qwen2.5:7b
   ```

3. **Memory Issues**
   ```bash
   # Use smaller model
   ollama pull qwen2.5:3b
   
   # Or adjust config.yaml
   generation:
     model: "qwen2.5:3b"
   ```

4. **Slow Embeddings**
   ```yaml
   # In config.yaml, use GPU if available
   embeddings:
     device: "cuda"
     batch_size: 64
   ```

### Debugging

Enable debug logging in `config.yaml`:

```yaml
logging:
  level: "DEBUG"
```

### Testing Components

```bash
# Test individual components
python scripts/test_components.py
python scripts/test_search.py
python scripts/test_without_ollama.py

# Test MCP server
mcp dev server.py
```

### Performance Tips

1. **GPU Acceleration**: Set `device: "cuda"` in config for embeddings
2. **Batch Processing**: Increase `batch_size` for faster indexing
3. **Model Size**: Use `qwen2.5:3b` for faster responses, `qwen2.5:14b` for better quality

## Best Practices

1. **Regular Updates**
   - Re-process books after major updates
   - Clear vector store cache monthly
   - Update embeddings when switching models

2. **Quality Control**
   - Always verify generated content against source
   - Use multiple sources for critical information
   - Include citations in production documents

3. **Resource Management**
   - Monitor API usage and costs
   - Implement rate limiting for production
   - Use local models when possible

## Support

For issues or questions:
1. Check existing issues on GitHub
2. Review the troubleshooting section
3. Submit a detailed bug report with:
   - Error messages
   - Steps to reproduce
   - System information