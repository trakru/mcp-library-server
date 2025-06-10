"""Embedding generation using sentence-transformers."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import torch
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", cache_dir: str = "./models/embeddings"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on ('cpu' or 'cuda')
            cache_dir: Directory to cache models and embeddings
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized embedding generator with {model_name} on {device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                cache_folder=str(self.cache_dir)
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            # Check cache first
            cached_embeddings = self._get_cached_embeddings(texts)
            if cached_embeddings is not None:
                logger.info(f"Using cached embeddings for {len(texts)} texts")
                return cached_embeddings
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for better similarity search
            )
            
            # Cache the embeddings
            self._cache_embeddings(texts, embeddings)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts."""
        # Create hash of all texts combined
        combined_text = "|||".join(texts)
        text_hash = hashlib.md5(combined_text.encode()).hexdigest()
        return f"{self.model_name}_{text_hash}"
    
    def _get_cached_embeddings(self, texts: List[str]) -> Union[np.ndarray, None]:
        """Try to get cached embeddings."""
        try:
            cache_key = self._get_cache_key(texts)
            cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Verify the texts match
                if cached_data['texts'] == texts:
                    return cached_data['embeddings']
            
        except Exception as e:
            logger.warning(f"Error loading cached embeddings: {e}")
        
        return None
    
    def _cache_embeddings(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Cache embeddings to disk."""
        try:
            cache_key = self._get_cache_key(texts)
            cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
            
            cache_data = {
                'texts': texts,
                'embeddings': embeddings,
                'model_name': self.model_name
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Error caching embeddings: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start + chunk_size - 100, start)
                search_end = min(end + 100, len(text))
                chunk_text = text[search_start:search_end]
                
                # Find the best break point (sentence end)
                sentence_ends = [i for i, char in enumerate(chunk_text) if char in '.!?']
                if sentence_ends:
                    # Use the sentence end closest to our target
                    target_pos = chunk_size - (search_start - start)
                    best_end = min(sentence_ends, key=lambda x: abs(x - target_pos))
                    end = search_start + best_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(end - chunk_overlap, start + 1)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def embed_chunked_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> tuple[List[str], np.ndarray]:
        """
        Chunk text and generate embeddings for each chunk.
        
        Args:
            text: Text to chunk and embed
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        embeddings = self.embed_texts(chunks)
        
        return chunks, embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(embedding1, embedding2))
    
    def find_most_similar(self, query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[tuple[int, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding
            embeddings: Array of embeddings to search
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if len(embeddings) == 0:
            return []
        
        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding.reshape(-1, 1)).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]