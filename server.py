#!/usr/bin/env python3
"""AI Book Agent MCP Server - Main entry point."""

import asyncio
import logging
from typing import Any, List, Optional
import json

from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions

# Import our modules
from src.utils.config import config
from src.parsers.epub_parser import EPUBParser
from src.embeddings.embeddings import EmbeddingGenerator
from src.generation.ollama_client import OllamaClient, DocumentGenerator
from src.search.vector_store import VectorStore, BookSearchEngine

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.get('logging.level', 'INFO')),
    format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

# Initialize components
try:
    # Initialize embedding generator
    logger.info("Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        model_name=config.embeddings['model'],
        device=config.embeddings['device'],
        cache_dir=config.embeddings['cache_dir']
    )
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        persist_directory=config.books['index_dir'],
        collection_name=config.vector_store['collection_name']
    )
    
    # Initialize search engine
    logger.info("Initializing search engine...")
    search_engine = BookSearchEngine(vector_store, embedding_generator)
    
    # Initialize Ollama client
    logger.info("Initializing Ollama client...")
    ollama_client = OllamaClient(
        base_url=config.generation['base_url'],
        model=config.generation['model']
    )
    
    # Initialize document generator
    logger.info("Initializing document generator...")
    doc_generator = DocumentGenerator(ollama_client)
    
    logger.info("All components initialized successfully!")
    
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

# Create MCP server
mcp = FastMCP("ai-book-agent")

@mcp.tool()
def search_books(query: str, max_results: int = 5, book_filter: Optional[List[str]] = None) -> str:
    """
    Search across ML textbooks for relevant content.
    
    Args:
        query: Search query (natural language)
        max_results: Maximum number of results to return (default: 5)
        book_filter: Optional list of book IDs to search within
        
    Returns:
        JSON string with search results containing content, similarity scores, and metadata
    """
    try:
        results = search_engine.search_books(
            query=query,
            max_results=max_results,
            book_filter=book_filter,
            similarity_threshold=config.search['similarity_threshold']
        )
        
        # Format results for better readability
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                "rank": i,
                "similarity": round(result["similarity"], 3),
                "book": f"{result['book_title']} by {result['author']}",
                "chapter": result['chapter_title'],
                "content": result['content'][:500] + "..." if len(result['content']) > 500 else result['content'],
                "full_content": result['content'],
                "book_id": result['book_id'],
                "chapter_id": result['chapter_id']
            }
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_books: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_chapter_content(book_id: str, chapter_id: Optional[int] = None, format: str = "markdown") -> str:
    """
    Get full content of a specific book chapter or entire book.
    
    Args:
        book_id: Unique identifier of the book
        chapter_id: Optional chapter number (if not provided, returns entire book)
        format: Output format - "markdown" or "plain" (default: markdown)
        
    Returns:
        Chapter or book content with metadata
    """
    try:
        content_data = search_engine.get_book_content(book_id, chapter_id)
        
        if "error" in content_data:
            return json.dumps(content_data)
        
        # Format content based on requested format
        if format == "markdown":
            formatted_content = f"# {content_data['book_title']}\n\n"
            formatted_content += f"**Author:** {content_data['author']}\n\n"
            
            if chapter_id is not None:
                formatted_content += f"## Chapter {chapter_id}: {content_data.get('chapter_title', '')}\n\n"
            
            formatted_content += content_data['content']
        else:
            formatted_content = content_data['content']
        
        result = {
            "book_id": book_id,
            "book_title": content_data['book_title'],
            "author": content_data['author'],
            "content": formatted_content,
            "metadata": {
                "chapter_id": content_data.get('chapter_id'),
                "chapter_title": content_data.get('chapter_title'),
                "chunk_count": content_data['chunk_count'],
                "word_count": content_data['word_count']
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_chapter_content: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def generate_section(topic: str, max_sources: int = 3, style: str = "technical", length: str = "standard") -> str:
    """
    Generate a documentation section based on book sources.
    
    Args:
        topic: Topic to write about
        max_sources: Maximum number of source passages to use (default: 3)
        style: Writing style - "technical", "tutorial", or "overview" (default: technical)
        length: Length preference - "brief", "standard", or "comprehensive" (default: standard)
        
    Returns:
        Generated documentation section with sources cited
    """
    try:
        # Search for relevant content
        search_results = search_engine.search_books(
            query=topic,
            max_results=max_sources,
            similarity_threshold=config.search['similarity_threshold']
        )
        
        if not search_results:
            return json.dumps({
                "error": f"No relevant content found for topic: {topic}"
            })
        
        # Extract source content
        source_content = [result['content'] for result in search_results]
        
        # Generate section
        generated_content = doc_generator.generate_section(
            topic=topic,
            source_content=source_content,
            style=style,
            max_length=length
        )
        
        # Prepare source citations
        sources = []
        for i, result in enumerate(search_results, 1):
            sources.append({
                "source_number": i,
                "book": f"{result['book_title']} by {result['author']}",
                "chapter": result['chapter_title'],
                "similarity": round(result['similarity'], 3)
            })
        
        result = {
            "topic": topic,
            "style": style,
            "length": length,
            "generated_content": generated_content,
            "sources_used": sources,
            "generation_metadata": {
                "total_sources": len(sources),
                "content_length": len(generated_content),
                "word_count": len(generated_content.split())
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in generate_section: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def explain_concept(concept: str, include_examples: bool = True, max_sources: int = 3) -> str:
    """
    Explain an ML concept using textbook sources.
    
    Args:
        concept: ML concept to explain
        include_examples: Whether to include practical examples (default: True)
        max_sources: Maximum number of sources to use (default: 3)
        
    Returns:
        Generated explanation with source citations
    """
    try:
        # Search for relevant content about the concept
        search_results = search_engine.search_books(
            query=f"{concept} definition explanation",
            max_results=max_sources,
            similarity_threshold=config.search['similarity_threshold']
        )
        
        if not search_results:
            return json.dumps({
                "error": f"No relevant content found for concept: {concept}"
            })
        
        # Extract source content
        source_content = [result['content'] for result in search_results]
        
        # Generate explanation
        explanation = doc_generator.explain_concept(
            concept=concept,
            source_content=source_content,
            include_examples=include_examples
        )
        
        # Prepare source citations
        sources = []
        for i, result in enumerate(search_results, 1):
            sources.append({
                "source_number": i,
                "book": f"{result['book_title']} by {result['author']}",
                "chapter": result['chapter_title'],
                "similarity": round(result['similarity'], 3)
            })
        
        result = {
            "concept": concept,
            "explanation": explanation,
            "include_examples": include_examples,
            "sources_used": sources,
            "metadata": {
                "total_sources": len(sources),
                "explanation_length": len(explanation),
                "word_count": len(explanation.split())
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in explain_concept: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def cite_sources(book_id: str, format: str = "APA") -> str:
    """
    Generate proper citations for book content.
    
    Args:
        book_id: Book identifier to cite
        format: Citation format - "APA", "MLA", or "Chicago" (default: APA)
        
    Returns:
        Formatted citation
    """
    try:
        # Get book metadata from a chunk
        chunks = vector_store.get_book_chunks(book_id)
        if not chunks:
            return json.dumps({"error": f"Book {book_id} not found"})
        
        metadata = chunks[0]["metadata"]
        title = metadata.get("book_title", "Unknown Title")
        author = metadata.get("author", "Unknown Author")
        
        # Generate citation based on format
        if format.upper() == "APA":
            citation = f"{author}. {title}."
        elif format.upper() == "MLA":
            citation = f"{author}. {title}."
        elif format.upper() == "CHICAGO":
            citation = f"{author}. {title}."
        else:
            citation = f"{author}. {title}."
        
        result = {
            "book_id": book_id,
            "format": format,
            "citation": citation,
            "metadata": {
                "title": title,
                "author": author
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in cite_sources: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def list_available_books() -> str:
    """
    List all available books in the knowledge base.
    
    Returns:
        JSON list of available books with metadata
    """
    try:
        stats = vector_store.get_stats()
        books_info = []
        
        for book_id in stats["books"]:
            chunks = vector_store.get_book_chunks(book_id)
            if chunks:
                metadata = chunks[0]["metadata"]
                book_info = {
                    "book_id": book_id,
                    "title": metadata.get("book_title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "total_chunks": len(chunks),
                    "total_words": sum(len(chunk["content"].split()) for chunk in chunks)
                }
                books_info.append(book_info)
        
        result = {
            "total_books": len(books_info),
            "total_chunks": stats["total_chunks"],
            "books": books_info
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in list_available_books: {e}")
        return json.dumps({"error": str(e)})

@mcp.resource("config://current")
def get_current_config() -> str:
    """Get current configuration settings."""
    return json.dumps({
        "embeddings": {
            "model": config.embeddings['model'],
            "device": config.embeddings['device']
        },
        "generation": {
            "model": config.generation['model'],
            "base_url": config.generation['base_url']
        },
        "search": {
            "max_results": config.search['max_results'],
            "similarity_threshold": config.search['similarity_threshold']
        },
        "vector_store": {
            "collection_name": config.vector_store['collection_name'],
            "chunk_size": config.vector_store['chunk_size']
        }
    }, indent=2)

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AI Book Agent MCP Server...")
    logger.info(f"Available books: {len(vector_store.list_books())}")
    logger.info(f"Total chunks in vector store: {vector_store.get_stats()['total_chunks']}")
    
    # The mcp.run() method will handle the server startup
    mcp.run()