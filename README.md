# AI Book Agent MCP Server

An MCP (Model Context Protocol) server that provides AI assistants with intelligent access to ML textbook content for creating accurate, source-grounded documentation. This pure Python implementation uses local models for complete privacy and cost efficiency.

## Overview

This project transforms authoritative ML textbooks into a knowledge service that any MCP-compatible AI assistant can access. Using local LLMs (Qwen) and embeddings (sentence-transformers), it creates a private, cost-effective RAG system exposed via the official Python MCP SDK.

## Why MCP?

### Traditional Approach Limitations
- Knowledge locked in a single application
- Users must switch between tools
- Cannot leverage AI assistants they already use
- Difficult to integrate with existing workflows

### MCP Server Benefits
- **Universal Access**: Works with Claude Desktop, VS Code, and any MCP client
- **Workflow Integration**: Use book knowledge directly in your IDE or chat
- **Composability**: Combine with other MCP servers (filesystem, GitHub, etc.)
- **Future-Proof**: As MCP ecosystem grows, your book agent automatically works with new tools

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Claude Desktop    │     │      VS Code        │
│  "Explain ML drift  │     │  "Generate docs     │
│   from textbooks"   │     │   for monitoring"   │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └─────────────┬─────────────┘
                         │ MCP Protocol (stdio/HTTP)
                         ▼
            ┌─────────────────────────────────────┐
            │     Python MCP Server               │
            │  (Official modelcontextprotocol/    │
            │         python-sdk)                 │
            ├─────────────────────────────────────┤
            │ MCP Tools:                          │
            │ - @mcp.tool() search_books()        │
            │ - @mcp.tool() get_chapter_content() │
            │ - @mcp.tool() generate_section()    │
            │ - @mcp.tool() cite_sources()        │
            ├─────────────────────────────────────┤
            │ RAG Components (same process):      │
            │ - EPUB Parser (ebooklib)            │
            │ - Embeddings (sentence-transformers)│
            │ - Vector Store (ChromaDB/FAISS)     │
            │ - LLM Generation (Ollama/Qwen)      │
            └─────────────────────────────────────┘
```

## Technology Stack

- **MCP Framework**: Official Python SDK (`mcp[cli]`)
- **Language**: Python 3.11+
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama with Qwen 2.5 7B
- **Vector Store**: ChromaDB/FAISS
- **Book Parsing**: ebooklib, BeautifulSoup4
- **Transport**: stdio (local) or HTTP (remote)

## Core Features as MCP Tools

### 1. **searchBooks**
Search across all indexed books for relevant content
```typescript
{
  name: "searchBooks",
  description: "Search ML textbooks for specific topics or concepts",
  inputSchema: {
    query: string,
    bookFilter?: string[],
    maxResults?: number,
    includeContext?: boolean
  }
}
```

### 2. **getChapterContent**
Retrieve specific chapter or section content
```typescript
{
  name: "getChapterContent",
  description: "Get full content of a specific book chapter",
  inputSchema: {
    bookId: string,
    chapterId: string,
    format?: "markdown" | "plain"
  }
}
```

### 3. **generateSection**
Generate documentation based on book content
```typescript
{
  name: "generateSection",
  description: "Generate documentation section grounded in textbook sources",
  inputSchema: {
    topic: string,
    outline?: string[],
    style?: "technical" | "tutorial" | "overview",
    maxSources?: number
  }
}
```

### 4. **citeSources**
Get proper citations for content
```typescript
{
  name: "citeSources",
  description: "Generate proper citations for book content",
  inputSchema: {
    bookId: string,
    pageNumbers?: number[],
    format?: "APA" | "MLA" | "Chicago"
  }
}
```

## Resources

The server exposes book content as browsable resources:

```
/books
├── /designing-ml-systems
│   ├── metadata.json
│   ├── /chapters
│   │   ├── /1-introduction
│   │   ├── /2-ml-systems-design
│   │   └── ...
│   └── /topics
│       ├── /monitoring
│       ├── /deployment
│       └── ...
└── /other-ml-book
    └── ...
```

## Prompts

Pre-configured prompts for common tasks:

### doc_generator
```yaml
name: doc_generator
description: Generate technical documentation from book sources
arguments:
  - name: topic
    description: The topic to document
  - name: audience
    description: Target audience (beginner/intermediate/advanced)
  - name: length
    description: Desired length (brief/standard/comprehensive)
```

### concept_explainer
```yaml
name: concept_explainer
description: Explain ML concepts using textbook definitions
arguments:
  - name: concept
    description: The concept to explain
  - name: include_examples
    description: Whether to include practical examples
```

## Use Cases

### 1. In Claude Desktop
```
User: "Explain model drift using the ML textbooks"
Claude: [Uses searchBooks tool to find drift content]
        [Retrieves relevant chapters]
        [Generates explanation with citations]
```

### 2. In VS Code
```python
# User comment: "TODO: Add monitoring based on best practices"
# AI Assistant uses book agent to generate monitoring code
```

### 3. Documentation Pipeline
```bash
# Automated doc generation using MCP tools
mcp-client generate-docs \
  --server ai-book-agent \
  --topics "deployment,monitoring,testing" \
  --output ml-best-practices.md
```

## Getting Started

### Prerequisites

- Linux system (Ubuntu 22.04+ recommended)
- Python 3.11+
- Node.js 18+
- 16GB RAM minimum
- 20GB free disk space

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-book-agent

# Install Python dependencies (with MCP SDK)
pip install "mcp[cli]" sentence-transformers chromadb ollama ebooklib beautifulsoup4

# Or use requirements.txt
pip install -r requirements.txt

# Install Ollama for local LLM
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull qwen2.5:7b

# Index existing books
python scripts/index_books.py
```

### Configuration

**Configure the server:**
```yaml
# config.yaml
embeddings:
  model: "all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda"
  
generation:
  provider: "ollama"
  model: "qwen2.5:7b"
  base_url: "http://localhost:11434"
  
books:
  data_dir: "data/epub"
  index_dir: "data/vector_db"
```
**Add to Claude Desktop config:**
```json
{
  "mcpServers": {
    "ai-book-agent": {
      "command": "python",
      "args": ["/path/to/ai-book-agent/server.py"]
    }
  }
}
```

### Basic Usage

Once configured, the tools are available in any MCP client:

```
You: What does the <author> say about feature engineering?
Assistant: I'll search the ML textbooks for information about feature engineering.

[Calling searchBooks with query="feature engineering"]
[Found 5 relevant sections in "<Book Title>"]

According to <author> in "<Book Title>":

Feature engineering is described as... [content with citations]
```

## Development

### Adding New Books

```bash
# Place EPUB in data directory
cp new-ml-book.epub data/epub/

# Re-index all books
python scripts/index_books.py

# Server will automatically pick up new content
```

### Extending Tools

Add new tools directly in `server.py`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-book-agent")

@mcp.tool()
def compare_approaches(approach1: str, approach2: str) -> str:
    """Compare different ML approaches from multiple books"""
    results1 = search_books(approach1, 3)
    results2 = search_books(approach2, 3)
    
    comparison = generate_comparison(results1, results2)
    return comparison

@mcp.resource("book://{book_id}/summary")
def get_book_summary(book_id: str) -> str:
    """Get a summary of a specific book"""
    return load_book_summary(book_id)
```

## Testing

```bash
# Test MCP server locally
mcp dev server.py

# Test individual components
python scripts/test_components.py
python scripts/test_search.py

# Run full test suite
pytest tests/

# Test with Claude Desktop
mcp install server.py
```

## Project Structure

```
ai-book-agent/
├── server.py             # Main MCP server (entry point)
├── src/                  # Python modules
│   ├── parsers/          # EPUB parsing
│   ├── embeddings/       # Embedding generation
│   ├── search/           # Vector search & retrieval
│   ├── generation/       # LLM integration (Ollama)
│   └── utils/            # Configuration and helpers
├── scripts/              # Utility scripts
│   ├── index_books.py    # Index EPUB files
│   ├── test_*.py         # Test individual components
│   └── setup.py          # Initial setup
├── data/
│   ├── epub/             # Source EPUB files
│   ├── processed/        # Processed book content
│   └── vector_db/        # Vector store data
├── tests/                # Test suites
├── config.yaml           # Configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## Pure Python Architecture

This project uses a simplified, single-language approach:
- **Python MCP Server**: Official SDK handles MCP protocol and tool exposure
- **Integrated RAG**: All ML components run in the same Python process
- **Local Models**: Complete privacy with Ollama and sentence-transformers

Benefits of this approach:
- **Simpler deployment**: Single Python service
- **Direct ML access**: No API overhead between MCP and RAG
- **Easier debugging**: One codebase, one process
- **Better performance**: No network calls between components

## Remote Access

For accessing the server from anywhere:

```bash
# Option 1: Cloudflare Tunnel (recommended)
cloudflared tunnel create book-agent
cloudflared tunnel run --url http://localhost:8080 book-agent

# Option 2: Configure with your domain
# See USER_GUIDE.md for detailed setup
```

## Roadmap

- [x] Basic MCP server structure
- [x] EPUB parsing and indexing
- [x] Core search tools
- [ ] Advanced RAG features
- [ ] Multi-book cross-referencing
- [ ] PDF support
- [ ] Streaming responses for long content
- [ ] Caching layer for performance
- [ ] Book update notifications

## Contributing

See [USER_GUIDE.md](USER_GUIDE.md) for details on:
- Adding new tools
- Improving search algorithms
- Supporting new book formats
- Performance optimizations