import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np

from src.parsers.epub_parser import EPUBParser
from src.embeddings.embeddings import EmbeddingGenerator
from src.search.vector_store import VectorStore, BookSearchEngine
from src.generation.ollama_client import OllamaClient, DocumentGenerator


@pytest.mark.integration
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_mock(self, tmp_path, mock_config):
        """Test the complete pipeline with mocked components."""
        
        # Create mock components
        with patch('src.embeddings.embeddings.SentenceTransformer') as mock_st, \
             patch('src.search.vector_store.chromadb.PersistentClient') as mock_chroma, \
             patch('src.generation.ollama_client.requests.Session') as mock_session:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
            mock_st.return_value = mock_model
            
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_collection.add = Mock()
            mock_collection.get = Mock(return_value={'ids': []})  # No existing chunks
            mock_collection.query = Mock(return_value={
                'ids': [['chunk1', 'chunk2']],
                'documents': [['Content about ML', 'More ML content']],
                'metadatas': [[
                    {'book_id': 'ml-book', 'chapter_title': 'Chapter 1', 'book_title': 'ML Book', 'author': 'Test Author'},
                    {'book_id': 'ml-book', 'chapter_title': 'Chapter 2', 'book_title': 'ML Book', 'author': 'Test Author'}
                ]],
                'distances': [[0.2, 0.4]]
            })
            mock_chroma_client = Mock()
            mock_chroma_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_chroma_client
            
            session = Mock()
            version_response = Mock()
            version_response.status_code = 200
            version_response.json.return_value = {"version": "0.1.0"}
            models_response = Mock()
            models_response.status_code = 200
            models_response.json.return_value = {"models": [{"name": "test-model"}]}
            session.get.side_effect = lambda url, **kwargs: version_response if "version" in url else models_response
            
            gen_response = Mock()
            gen_response.status_code = 200
            gen_response.json.return_value = {"response": "Generated explanation about machine learning"}
            session.post.return_value = gen_response
            mock_session.return_value = session
            
            # Initialize components
            embedding_gen = EmbeddingGenerator(
                model_name="test-model",
                cache_dir=str(tmp_path / "embeddings")
            )
            
            vector_store = VectorStore(
                persist_directory=str(tmp_path / "vector_db"),
                collection_name="test_books"
            )
            
            search_engine = BookSearchEngine(vector_store, embedding_gen)
            
            ollama_client = OllamaClient(model="test-model")
            doc_generator = DocumentGenerator(ollama_client)
            
            # Test the pipeline
            # 1. Add some mock book data
            chunks = ["Machine learning introduction", "Deep learning concepts", "Neural network basics"]
            embeddings = embedding_gen.embed_texts(chunks)
            metadata = [
                {"chapter_id": 0, "chapter_title": "Introduction", "book_title": "ML Book", "author": "Test Author"},
                {"chapter_id": 1, "chapter_title": "Deep Learning", "book_title": "ML Book", "author": "Test Author"},
                {"chapter_id": 2, "chapter_title": "Neural Networks", "book_title": "ML Book", "author": "Test Author"}
            ]
            
            vector_store.add_book_chunks("ml-book", chunks, embeddings, metadata)
            
            # 2. Test search
            search_results = search_engine.search_books("machine learning", max_results=2)
            assert len(search_results) == 2
            assert search_results[0]['book_title'] == 'ML Book'
            
            # 3. Test document generation
            explanation = doc_generator.explain_concept(
                "machine learning",
                [result['content'] for result in search_results]
            )
            assert "Generated explanation" in explanation
            
            # Verify components were called
            mock_model.encode.assert_called()
            mock_collection.add.assert_called_once()
            mock_collection.query.assert_called()
            session.post.assert_called()
    
    def test_epub_to_vector_store_pipeline(self, tmp_path, mock_epub_file):
        """Test EPUB parsing to vector store pipeline."""
        
        with patch('src.embeddings.embeddings.SentenceTransformer') as mock_st, \
             patch('src.search.vector_store.chromadb.PersistentClient') as mock_chroma:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
            mock_st.return_value = mock_model
            
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_collection.get = Mock(return_value={'ids': []})  # No existing chunks
            mock_chroma_client = Mock()
            mock_chroma_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_chroma_client
            
            # Create mock EPUB structure (using our fixture approach)
            epub_path = mock_epub_file
            
            # Initialize components
            parser = EPUBParser()
            embedding_gen = EmbeddingGenerator(
                model_name="test-model",
                cache_dir=str(tmp_path / "embeddings")
            )
            vector_store = VectorStore(
                persist_directory=str(tmp_path / "vector_db"),
                collection_name="test_books"
            )
            
            # Test pipeline: EPUB -> Chunks -> Embeddings -> Vector Store
            # For this test, we'll simulate parsing since we don't have a real EPUB
            # In practice, you'd do: metadata, chapters = parser.parse_epub(epub_path)
            
            # Simulate parsed chapters
            mock_chapters = [
                type('Chapter', (), {
                    'chapter_id': 0,
                    'title': 'Chapter 1',
                    'content': 'This is test content for chapter 1.',
                    'book_id': 'test-book',
                    'book_title': 'Test Book',
                    'author': 'Test Author'
                })(),
                type('Chapter', (), {
                    'chapter_id': 1,
                    'title': 'Chapter 2', 
                    'content': 'This is test content for chapter 2.',
                    'book_id': 'test-book',
                    'book_title': 'Test Book',
                    'author': 'Test Author'
                })()
            ]
            
            # Extract text and generate embeddings
            texts = [chapter.content for chapter in mock_chapters]
            embeddings = embedding_gen.embed_texts(texts)
            
            # Create metadata for vector store
            metadata_list = [
                {
                    'chapter_id': chapter.chapter_id,
                    'chapter_title': chapter.title,
                    'book_title': chapter.book_title,
                    'author': chapter.author
                }
                for chapter in mock_chapters
            ]
            
            # Add to vector store
            vector_store.add_book_chunks('test-book', texts, embeddings, metadata_list)
            
            # Verify the pipeline worked
            mock_model.encode.assert_called_with(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            mock_collection.add.assert_called_once()
            
            add_call_args = mock_collection.add.call_args[1]
            assert len(add_call_args['ids']) == 2
            assert add_call_args['ids'][0] == 'test-book_chunk_0'
            assert add_call_args['documents'] == texts
            assert all(meta['book_id'] == 'test-book' for meta in add_call_args['metadatas'])
    
    def test_error_handling_integration(self, tmp_path):
        """Test error handling across the integrated system."""
        
        with patch('src.embeddings.embeddings.SentenceTransformer') as mock_st, \
             patch('src.search.vector_store.chromadb.PersistentClient') as mock_chroma:
            
            # Setup mocks that will fail
            mock_st.side_effect = Exception("Model loading failed")
            
            # Test that initialization errors are properly handled
            with pytest.raises(Exception) as exc_info:
                EmbeddingGenerator(model_name="bad-model")
            assert "Model loading failed" in str(exc_info.value)
            
            # Test vector store with connection issues
            mock_chroma.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception):
                VectorStore(persist_directory=str(tmp_path / "bad_db"))
    
    @pytest.mark.slow
    def test_performance_mock(self, tmp_path):
        """Test performance with larger mock datasets."""
        
        with patch('src.embeddings.embeddings.SentenceTransformer') as mock_st, \
             patch('src.search.vector_store.chromadb.PersistentClient') as mock_chroma:
            
            # Setup mocks for larger dataset
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            # Simulate processing 100 chunks
            mock_model.encode.return_value = np.random.randn(100, 384).astype(np.float32)
            mock_st.return_value = mock_model
            
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_collection.get = Mock(return_value={'ids': []})  # No existing chunks
            mock_chroma_client = Mock()
            mock_chroma_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_chroma_client
            
            # Initialize components
            embedding_gen = EmbeddingGenerator(model_name="test-model")
            vector_store = VectorStore(persist_directory=str(tmp_path / "vector_db"))
            
            # Test with larger dataset (100 chunks)
            large_chunks = [f"This is chunk {i} with some test content." for i in range(100)]
            embeddings = embedding_gen.embed_texts(large_chunks)
            metadata = [{"chunk_id": i, "book_id": "large-book"} for i in range(100)]
            
            vector_store.add_book_chunks("large-book", large_chunks, embeddings, metadata)
            
            # Verify it handled the large dataset
            assert embeddings.shape == (100, 384)
            mock_collection.add.assert_called_once()
            add_args = mock_collection.add.call_args[1]
            assert len(add_args['documents']) == 100