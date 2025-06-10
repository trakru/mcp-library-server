# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Book Agent MCP Server - A pure Python system that exposes ML textbook knowledge via MCP protocol using the official Python SDK. Uses local models (Ollama/Qwen) for privacy and cost efficiency.

## Architecture

- **Single Python Service**: MCP server and RAG components in one process
- **Official MCP SDK**: Uses `modelcontextprotocol/python-sdk`
- **Local Models**: Ollama for generation, sentence-transformers for embeddings
- **No external APIs**: Fully self-contained system

## Common Development Commands

```bash
# Setup
pip install "mcp[cli]" sentence-transformers chromadb ollama ebooklib beautifulsoup4

# Index books
python scripts/index_books.py

# Development mode (with auto-reload)
mcp dev server.py

# Production
python server.py

# Testing
python scripts/test_components.py
python scripts/test_search.py
pytest tests/
```

## Configuration

Key configuration file:
- `config.yaml`: All settings (embeddings, generation, search, logging)

## Local Models

- **Embeddings**: all-MiniLM-L6-v2 (via sentence-transformers)
- **Generation**: Qwen 2.5 7B (via Ollama)
- **No cloud API dependencies** - fully local operation

## MCP Tools

1. `searchBooks`: Semantic search across books
2. `getChapterContent`: Retrieve specific sections
3. `generateSection`: Create documentation from sources
4. `citeSources`: Generate proper citations

## Testing Queries

Common test queries to verify the system:
- "What are ML monitoring best practices?"
- "Explain feature engineering for time series"
- "How to handle data drift in production?"

## Deployment

For remote access:
1. Cloudflare Tunnel: `cloudflared tunnel run --url http://localhost:8080 book-agent`
2. Configure Claude Desktop to use remote MCP endpoint