import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pickle
import hashlib

from src.embeddings.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for the EmbeddingGenerator class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        return mock_model
    
    @pytest.fixture
    def embedding_generator(self, mock_sentence_transformer, tmp_path):
        """Create EmbeddingGenerator with mocked model."""
        with patch('src.embeddings.embeddings.SentenceTransformer', return_value=mock_sentence_transformer):
            generator = EmbeddingGenerator(
                model_name="test-model",
                device="cpu",
                cache_dir=str(tmp_path / "cache")
            )
            return generator
    
    def test_initialization(self, tmp_path):
        """Test EmbeddingGenerator initialization."""
        with patch('src.embeddings.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator(
                model_name="test-model",
                device="cpu",
                cache_dir=str(tmp_path / "cache")
            )
            
            assert generator.model_name == "test-model"
            assert generator.device == "cpu"
            assert generator.embedding_dim == 384
            assert generator.cache_dir.exists()
    
    def test_load_model_error(self, tmp_path):
        """Test model loading error handling."""
        with patch('src.embeddings.embeddings.SentenceTransformer', side_effect=Exception("Model loading failed")):
            with pytest.raises(Exception) as exc_info:
                EmbeddingGenerator(
                    model_name="invalid-model",
                    device="cpu",
                    cache_dir=str(tmp_path / "cache")
                )
            assert "Model loading failed" in str(exc_info.value)
    
    def test_embed_text_single(self, embedding_generator):
        """Test embedding a single text."""
        test_text = "This is a test sentence."
        expected_embedding = np.random.randn(384).astype(np.float32)
        embedding_generator.model.encode.return_value = np.array([expected_embedding])
        
        result = embedding_generator.embed_text(test_text)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        embedding_generator.model.encode.assert_called_once()
    
    def test_embed_texts_multiple(self, embedding_generator):
        """Test embedding multiple texts."""
        test_texts = ["First sentence.", "Second sentence.", "Third sentence."]
        expected_embeddings = np.random.randn(3, 384).astype(np.float32)
        embedding_generator.model.encode.return_value = expected_embeddings
        
        result = embedding_generator.embed_texts(test_texts, batch_size=2, show_progress=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)
        embedding_generator.model.encode.assert_called_with(
            test_texts,
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    def test_embed_texts_empty(self, embedding_generator):
        """Test embedding empty text list."""
        result = embedding_generator.embed_texts([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
    
    def test_caching_mechanism(self, embedding_generator, tmp_path):
        """Test embedding caching functionality."""
        test_texts = ["Test sentence for caching."]
        expected_embeddings = np.random.randn(1, 384).astype(np.float32)
        embedding_generator.model.encode.return_value = expected_embeddings
        
        # First call - should generate and cache
        result1 = embedding_generator.embed_texts(test_texts)
        assert embedding_generator.model.encode.call_count == 1
        
        # Reset mock
        embedding_generator.model.encode.reset_mock()
        
        # Second call - should use cache
        result2 = embedding_generator.embed_texts(test_texts)
        assert embedding_generator.model.encode.call_count == 0
        np.testing.assert_array_equal(result1, result2)
    
    def test_cache_key_generation(self, embedding_generator):
        """Test cache key generation."""
        texts1 = ["Text 1", "Text 2"]
        texts2 = ["Text 2", "Text 1"]  # Different order
        texts3 = ["Text 1", "Text 2"]  # Same as texts1
        
        key1 = embedding_generator._get_cache_key(texts1)
        key2 = embedding_generator._get_cache_key(texts2)
        key3 = embedding_generator._get_cache_key(texts3)
        
        assert key1 != key2  # Different order should produce different keys
        assert key1 == key3  # Same texts in same order should produce same key
    
    def test_chunk_text_simple(self, embedding_generator):
        """Test text chunking with simple case."""
        text = "This is a short text."
        chunks = embedding_generator.chunk_text(text, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_with_overlap(self, embedding_generator):
        """Test text chunking with overlap."""
        # Create a long text with clear sentence boundaries
        text = "First sentence. " * 50  # ~750 characters
        chunks = embedding_generator.chunk_text(text, chunk_size=200, chunk_overlap=50)
        
        assert len(chunks) > 1
        # Check that chunks have some overlap
        for i in range(len(chunks) - 1):
            # There should be some overlapping content
            assert len(chunks[i]) > 0
            assert len(chunks[i+1]) > 0
    
    def test_chunk_text_sentence_boundaries(self, embedding_generator):
        """Test that chunking respects sentence boundaries."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = embedding_generator.chunk_text(text, chunk_size=40, chunk_overlap=10)
        
        # Check that chunks end at sentence boundaries when possible
        for chunk in chunks:
            assert chunk.strip()  # No empty chunks
            # Most chunks should end with punctuation
            if chunk != chunks[-1]:  # Except possibly the last one
                assert chunk[-1] in '.!?' or len(chunk) < 40
    
    def test_embed_chunked_text(self, embedding_generator):
        """Test chunking and embedding text together."""
        text = "This is a test text. " * 20
        expected_chunks = ["chunk1", "chunk2", "chunk3"]
        expected_embeddings = np.random.randn(3, 384).astype(np.float32)
        
        with patch.object(embedding_generator, 'chunk_text', return_value=expected_chunks):
            embedding_generator.model.encode.return_value = expected_embeddings
            
            chunks, embeddings = embedding_generator.embed_chunked_text(text, chunk_size=100)
            
            assert chunks == expected_chunks
            assert embeddings.shape == (3, 384)
    
    def test_similarity_calculation(self, embedding_generator):
        """Test cosine similarity calculation."""
        # Create normalized vectors for easy testing
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])  # Same direction
        embedding3 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        embedding4 = np.array([-1.0, 0.0, 0.0])  # Opposite direction
        
        # Normalize
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        embedding3 = embedding3 / np.linalg.norm(embedding3)
        embedding4 = embedding4 / np.linalg.norm(embedding4)
        
        assert pytest.approx(embedding_generator.similarity(embedding1, embedding2), 0.001) == 1.0
        assert pytest.approx(embedding_generator.similarity(embedding1, embedding3), 0.001) == 0.0
        assert pytest.approx(embedding_generator.similarity(embedding1, embedding4), 0.001) == -1.0
    
    def test_find_most_similar(self, embedding_generator):
        """Test finding most similar embeddings."""
        # Create query embedding
        query_embedding = np.array([1.0, 0.0, 0.0])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Create embeddings with known similarities
        embeddings = np.array([
            [1.0, 0.0, 0.0],    # Most similar (same as query)
            [0.8, 0.6, 0.0],    # Second most similar
            [0.0, 1.0, 0.0],    # Orthogonal
            [-1.0, 0.0, 0.0],   # Opposite
        ])
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        results = embedding_generator.find_most_similar(query_embedding, embeddings, top_k=3)
        
        assert len(results) == 3
        # Check that results are sorted by similarity (descending)
        assert results[0][0] == 0  # Index of most similar
        assert results[0][1] > results[1][1]  # Similarity scores are descending
        assert results[1][1] > results[2][1]
    
    def test_find_most_similar_empty(self, embedding_generator):
        """Test finding similar embeddings with empty array."""
        query_embedding = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([])
        
        results = embedding_generator.find_most_similar(query_embedding, embeddings, top_k=5)
        assert results == []
    
    def test_error_handling_in_embed_texts(self, embedding_generator):
        """Test error handling in embed_texts method."""
        embedding_generator.model.encode.side_effect = Exception("Encoding failed")
        
        with pytest.raises(Exception) as exc_info:
            embedding_generator.embed_texts(["Test text"])
        assert "Encoding failed" in str(exc_info.value)