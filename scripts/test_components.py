#!/usr/bin/env python3
"""Test script for AI Book Agent components."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.embeddings.embeddings import EmbeddingGenerator
from src.generation.ollama_client import OllamaClient, DocumentGenerator
from src.search.vector_store import VectorStore, BookSearchEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embeddings():
    """Test embedding generation."""
    logger.info("Testing embedding generation...")
    
    try:
        embedding_generator = EmbeddingGenerator(
            model_name=config.embeddings['model'],
            device=config.embeddings['device'],
            cache_dir=config.embeddings['cache_dir']
        )
        
        # Test single embedding
        test_text = "Machine learning is a subset of artificial intelligence."
        embedding = embedding_generator.embed_text(test_text)
        logger.info(f"Single embedding shape: {embedding.shape}")
        
        # Test multiple embeddings
        test_texts = [
            "Deep learning uses neural networks.",
            "Random forests are ensemble methods.",
            "Support vector machines find optimal decision boundaries."
        ]
        embeddings = embedding_generator.embed_texts(test_texts)
        logger.info(f"Multiple embeddings shape: {embeddings.shape}")
        
        # Test chunking
        long_text = "This is a long text that will be chunked. " * 50
        chunks = embedding_generator.chunk_text(long_text, chunk_size=100, chunk_overlap=20)
        logger.info(f"Chunked into {len(chunks)} pieces")
        
        logger.info("✅ Embedding tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False

def test_ollama():
    """Test Ollama client."""
    logger.info("Testing Ollama client...")
    
    try:
        ollama_client = OllamaClient(
            base_url=config.generation['base_url'],
            model=config.generation['model']
        )
        
        # Test health check
        if not ollama_client.is_healthy():
            logger.error("Ollama server is not healthy")
            return False
        
        # Test simple generation
        prompt = "Explain what machine learning is in one sentence."
        response = ollama_client.generate(prompt, temperature=0.3, max_tokens=100)
        logger.info(f"Generated response: {response[:100]}...")
        
        # Test document generator
        doc_generator = DocumentGenerator(ollama_client)
        
        sources = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "It is a branch of artificial intelligence based on the idea that systems can learn from data."
        ]
        
        section = doc_generator.generate_section(
            topic="Introduction to Machine Learning",
            source_content=sources,
            style="tutorial"
        )
        logger.info(f"Generated section length: {len(section)} characters")
        
        logger.info("✅ Ollama tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama test failed: {e}")
        return False

def test_vector_store():
    """Test vector store operations."""
    logger.info("Testing vector store...")
    
    try:
        # Initialize components
        embedding_generator = EmbeddingGenerator(
            model_name=config.embeddings['model'],
            device=config.embeddings['device']
        )
        
        vector_store = VectorStore(
            persist_directory=config.books['index_dir'],
            collection_name="test_collection"
        )
        
        # Test data
        test_chunks = [
            "Neural networks are computational models inspired by biological neural networks.",
            "Decision trees are a type of supervised learning algorithm.",
            "Clustering is an unsupervised learning technique for grouping data."
        ]
        
        test_metadata = [
            {"book_id": "test_book", "chapter_id": 1, "book_title": "Test Book", "author": "Test Author"},
            {"book_id": "test_book", "chapter_id": 2, "book_title": "Test Book", "author": "Test Author"},
            {"book_id": "test_book", "chapter_id": 3, "book_title": "Test Book", "author": "Test Author"}
        ]
        
        # Generate embeddings
        embeddings = embedding_generator.embed_texts(test_chunks)
        
        # Add to vector store
        vector_store.add_book_chunks("test_book", test_chunks, embeddings, test_metadata)
        logger.info("Added test chunks to vector store")
        
        # Test search
        query_embedding = embedding_generator.embed_text("What are neural networks?")
        results = vector_store.search(query_embedding, n_results=2)
        logger.info(f"Search returned {len(results)} results")
        
        # Test search engine
        search_engine = BookSearchEngine(vector_store, embedding_generator)
        search_results = search_engine.search_books("neural networks", max_results=2)
        logger.info(f"Search engine returned {len(search_results)} results")
        
        # Clean up
        vector_store.remove_book("test_book")
        logger.info("Cleaned up test data")
        
        logger.info("✅ Vector store tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False

def test_existing_books():
    """Test with existing books in the vector store."""
    logger.info("Testing with existing books...")
    
    try:
        # Initialize components
        embedding_generator = EmbeddingGenerator(
            model_name=config.embeddings['model'],
            device=config.embeddings['device']
        )
        
        vector_store = VectorStore(
            persist_directory=config.books['index_dir'],
            collection_name=config.vector_store['collection_name']
        )
        
        search_engine = BookSearchEngine(vector_store, embedding_generator)
        
        # Get stats
        stats = vector_store.get_stats()
        logger.info(f"Vector store contains {stats['total_chunks']} chunks from {stats['total_books']} books")
        
        if stats['total_books'] == 0:
            logger.warning("No books found in vector store. Run index_books.py first.")
            return True
        
        # Test search with existing data
        test_queries = [
            "machine learning monitoring",
            "feature engineering",
            "model deployment",
            "data drift"
        ]
        
        for query in test_queries:
            results = search_engine.search_books(query, max_results=3)
            logger.info(f"Query '{query}': {len(results)} results")
            if results:
                logger.info(f"  Top result: {results[0]['book_title']} (similarity: {results[0]['similarity']:.3f})")
        
        logger.info("✅ Existing books test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Existing books test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting component tests...\n")
    
    tests = [
        ("Embeddings", test_embeddings),
        ("Ollama", test_ollama),
        ("Vector Store", test_vector_store),
        ("Existing Books", test_existing_books)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()