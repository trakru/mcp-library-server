# AI Book Agent

An MCP (Model Context Protocol) server that provides ML textbook knowledge to Claude Desktop using fully local models - no external API keys required.

## Project Overview

This project enables Claude Desktop to access authoritative ML textbook knowledge through a local MCP server. Built with privacy in mind, it uses local models for both embeddings and text generation, ensuring your queries and book content never leave your machine.

## Problem Statement

- Organizations need accurate ML documentation based on established best practices
- Authoritative textbooks contain valuable knowledge but are difficult to search through
- External AI APIs raise privacy concerns for sensitive projects
- Need for source-grounded responses with proper citations

## Solution

An MCP server that:
1. **Processes** ML textbooks locally (EPUB format)
2. **Indexes** content using local sentence transformers
3. **Provides** semantic search through MCP tools
4. **Generates** responses using local Ollama models
5. **Integrates** seamlessly with Claude Desktop

## Core Features

- **MCP Integration**: Works directly with Claude Desktop via Model Context Protocol
- **Local Models**: Sentence transformers + Ollama (no API keys needed)
- **Privacy-First**: All processing happens locally on your machine
- **Source Attribution**: Always provides citations from source books
- **Fast Setup**: Automated installation with UV package manager
- **Extensible**: Easy to add new books and customize behavior

## Use Cases

1. **Technical Documentation**: Generate docs grounded in textbook best practices
2. **Knowledge Lookup**: Answer ML questions with authoritative sources
3. **Concept Explanation**: Get detailed explanations from multiple textbooks
4. **Best Practices**: Access established methodologies from expert authors

## Current Status

- ✅ MCP server with 4 core tools
- ✅ Local EPUB processing and vector indexing
- ✅ Sentence transformers for embeddings
- ✅ Ollama integration for text generation
- ✅ ChromaDB vector storage
- ✅ Automated setup with UV

## Technology Stack

- **Language**: Python 3.11+
- **MCP Framework**: Official Python SDK
- **Embeddings**: sentence-transformers (local)
- **Generation**: Ollama + Qwen 2.5 7B (local)
- **Vector Store**: ChromaDB
- **Package Management**: UV

## Getting Started

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-book-agent

# Run automated setup (installs everything)
./setup-uv.sh
```

### Manual Setup

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip sync requirements.txt

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b

# Process books
python scripts/index_books.py
```

### Configure Claude Desktop

Add to Claude Desktop config:

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

## Project Structure

```
ai-book-agent/
├── src/              # Core MCP server code
├── data/             # EPUB files and processed data
├── scripts/          # Setup and utility scripts
├── config.yaml       # Single configuration file
├── server.py         # MCP server entry point
├── setup-uv.sh       # Automated setup script
└── pyproject.toml    # Modern Python project config
```

## MCP Tools

The server provides these tools to Claude:

1. **searchBooks**: Semantic search across all indexed books
2. **getChapterContent**: Retrieve specific book sections
3. **generateSection**: Create documentation from sources
4. **citeSources**: Generate proper citations

## Contributing

See [USER_GUIDE_STANDALONE.md](USER_GUIDE_STANDALONE.md) for detailed instructions on:
- Adding new books to the system
- Extending the MCP server
- Testing and validation
- Best practices for development

## Requirements

- **OS**: macOS, Linux, or Windows
- **Python**: 3.11+
- **Memory**: 8GB RAM minimum
- **Storage**: 10GB free space

## License

[Your license here]

## Acknowledgments

Built to leverage authoritative ML textbooks including:
- "Designing Machine Learning Systems" by Chip Huyen
- [Additional books as you add them]