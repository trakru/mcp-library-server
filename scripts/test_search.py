#!/usr/bin/env python3
"""Quick search test with different thresholds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.embeddings.embeddings import EmbeddingGenerator
from src.search.vector_store import VectorStore, BookSearchEngine

def test_search_thresholds():
    """Test search with different similarity thresholds."""
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
    
    # Test queries
    queries = ["machine learning", "deployment", "monitoring", "data quality"]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        for threshold in thresholds:
            results = search_engine.search_books(
                query=query,
                max_results=3,
                similarity_threshold=threshold
            )
            
            if results:
                top_similarity = results[0]['similarity']
                print(f"Threshold {threshold}: {len(results)} results (top: {top_similarity:.3f})")
            else:
                print(f"Threshold {threshold}: 0 results")
        
        # Show top result with new threshold
        results = search_engine.search_books(query=query, max_results=1)
        if results:
            r = results[0]
            print(f"\nTop result:")
            print(f"  Book: {r['book_title']}")
            print(f"  Chapter: {r['chapter_title']}")
            print(f"  Similarity: {r['similarity']:.3f}")
            print(f"  Preview: {r['content'][:150]}...")

if __name__ == "__main__":
    test_search_thresholds()