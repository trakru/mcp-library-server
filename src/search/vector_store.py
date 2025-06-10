"""Vector store for semantic search using ChromaDB."""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for storing and searching book content embeddings."""
    
    def __init__(self, persist_directory: str, collection_name: str = "ml_books"):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "ML textbook content embeddings"}
        )
        
        logger.info(f"Initialized vector store at {persist_directory}")
        logger.info(f"Collection '{collection_name}' has {self.collection.count()} documents")
    
    def add_book_chunks(self, book_id: str, chunks: List[str], embeddings: np.ndarray, 
                       metadata_list: List[Dict[str, Any]]) -> None:
        """
        Add book chunks to the vector store.
        
        Args:
            book_id: Unique book identifier
            chunks: List of text chunks
            embeddings: Corresponding embeddings for chunks
            metadata_list: List of metadata dictionaries for each chunk
        """
        if len(chunks) != len(embeddings) or len(chunks) != len(metadata_list):
            raise ValueError("Chunks, embeddings, and metadata must have the same length")
        
        # Generate unique IDs for each chunk
        chunk_ids = [f"{book_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Prepare embeddings (ChromaDB expects list of lists)
        embeddings_list = embeddings.tolist()
        
        # Add book_id to each metadata entry
        for metadata in metadata_list:
            metadata["book_id"] = book_id
        
        try:
            # Remove any existing chunks for this book
            self.remove_book(book_id)
            
            # Add new chunks
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings_list,
                metadatas=metadata_list
            )
            
            logger.info(f"Added {len(chunks)} chunks for book {book_id}")
            
        except Exception as e:
            logger.error(f"Error adding chunks for book {book_id}: {e}")
            raise
    
    def remove_book(self, book_id: str) -> None:
        """
        Remove all chunks for a specific book.
        
        Args:
            book_id: Book identifier to remove
        """
        try:
            # Get all chunk IDs for this book
            results = self.collection.get(
                where={"book_id": book_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Removed {len(results['ids'])} chunks for book {book_id}")
            
        except Exception as e:
            logger.error(f"Error removing book {book_id}: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5,
               book_filter: Optional[List[str]] = None,
               similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            book_filter: Optional list of book IDs to filter by
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Prepare query embedding
            query_embedding_list = query_embedding.tolist()
            
            # Prepare where filter
            where_filter = {}
            if book_filter:
                where_filter["book_id"] = {"$in": book_filter}
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity (ChromaDB uses L2 distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= similarity_threshold:
                        search_results.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity": similarity,
                            "rank": i + 1
                        })
            
            logger.info(f"Found {len(search_results)} results above threshold {similarity_threshold}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def get_book_chunks(self, book_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific book.
        
        Args:
            book_id: Book identifier
            
        Returns:
            List of chunks with metadata
        """
        try:
            results = self.collection.get(
                where={"book_id": book_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    chunks.append({
                        "content": doc,
                        "metadata": metadata
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for book {book_id}: {e}")
            raise
    
    def list_books(self) -> List[str]:
        """
        List all book IDs in the store.
        
        Returns:
            List of unique book IDs
        """
        try:
            results = self.collection.get(include=["metadatas"])
            
            book_ids = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'book_id' in metadata:
                        book_ids.add(metadata['book_id'])
            
            return sorted(list(book_ids))
            
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            total_chunks = self.collection.count()
            books = self.list_books()
            
            stats = {
                "total_chunks": total_chunks,
                "total_books": len(books),
                "books": books,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "ML textbook content embeddings"}
            )
            
            logger.info("Cleared vector store")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise


class BookSearchEngine:
    """High-level search engine for books."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator):
        """
        Initialize search engine.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)
    
    def search_books(self, query: str, max_results: int = 5,
                    book_filter: Optional[List[str]] = None,
                    similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search books with natural language query.
        
        Args:
            query: Natural language search query
            max_results: Maximum number of results
            book_filter: Optional list of book IDs to search within
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_text(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=max_results,
                book_filter=book_filter,
                similarity_threshold=similarity_threshold
            )
            
            # Enhance results with additional information
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    "content": result["content"],
                    "similarity": result["similarity"],
                    "rank": result["rank"],
                    "book_id": result["metadata"].get("book_id"),
                    "book_title": result["metadata"].get("book_title"),
                    "author": result["metadata"].get("author"),
                    "chapter_title": result["metadata"].get("chapter_title"),
                    "chapter_id": result["metadata"].get("chapter_id"),
                    "word_count": len(result["content"].split())
                }
                enhanced_results.append(enhanced_result)
            
            self.logger.info(f"Search for '{query}' returned {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error searching for '{query}': {e}")
            raise
    
    def get_book_content(self, book_id: str, chapter_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get content for a specific book or chapter.
        
        Args:
            book_id: Book identifier
            chapter_id: Optional chapter identifier
            
        Returns:
            Book/chapter content with metadata
        """
        try:
            chunks = self.vector_store.get_book_chunks(book_id)
            
            if not chunks:
                return {"error": f"No content found for book {book_id}"}
            
            # Filter by chapter if specified
            if chapter_id is not None:
                chunks = [chunk for chunk in chunks 
                         if chunk["metadata"].get("chapter_id") == chapter_id]
                
                if not chunks:
                    return {"error": f"No content found for chapter {chapter_id} in book {book_id}"}
            
            # Combine content and extract metadata
            content_parts = [chunk["content"] for chunk in chunks]
            combined_content = "\n\n".join(content_parts)
            
            # Get metadata from first chunk
            metadata = chunks[0]["metadata"]
            
            result = {
                "book_id": book_id,
                "book_title": metadata.get("book_title"),
                "author": metadata.get("author"),
                "content": combined_content,
                "chunk_count": len(chunks),
                "word_count": len(combined_content.split())
            }
            
            if chapter_id is not None:
                result["chapter_id"] = chapter_id
                result["chapter_title"] = metadata.get("chapter_title")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting content for book {book_id}: {e}")
            raise
    
    def find_related_content(self, content: str, max_results: int = 3,
                           exclude_book: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find content related to the given text.
        
        Args:
            content: Text content to find related material for
            max_results: Maximum number of results
            exclude_book: Optional book ID to exclude from results
            
        Returns:
            List of related content
        """
        try:
            # Get all books except the excluded one
            all_books = self.vector_store.list_books()
            if exclude_book and exclude_book in all_books:
                all_books.remove(exclude_book)
            
            # Search for related content
            results = self.search_books(
                query=content,
                max_results=max_results,
                book_filter=all_books if exclude_book else None,
                similarity_threshold=0.6
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding related content: {e}")
            raise