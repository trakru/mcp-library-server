import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.search.vector_store import VectorStore, BookSearchEngine


class TestVectorStore:
    """Test suite for the VectorStore class."""
    
    @pytest.fixture
    def mock_chromadb_collection(self):
        """Create a mock ChromaDB collection."""
        collection = MagicMock()
        collection.count.return_value = 0
        collection.add = MagicMock()
        collection.delete = MagicMock()
        collection.get = MagicMock(return_value={'ids': [], 'documents': [], 'metadatas': []})
        collection.query = MagicMock(return_value={
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        })
        return collection
    
    @pytest.fixture
    def mock_chromadb_client(self, mock_chromadb_collection):
        """Create a mock ChromaDB client."""
        client = Mock()
        client.get_or_create_collection.return_value = mock_chromadb_collection
        client.delete_collection = MagicMock()
        return client
    
    @pytest.fixture
    def vector_store(self, tmp_path, mock_chromadb_client):
        """Create VectorStore with mocked ChromaDB."""
        with patch('src.search.vector_store.chromadb.PersistentClient', return_value=mock_chromadb_client):
            store = VectorStore(
                persist_directory=str(tmp_path / "test_vector_db"),
                collection_name="test_collection"
            )
            return store
    
    def test_initialization(self, tmp_path):
        """Test VectorStore initialization."""
        with patch('src.search.vector_store.chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_collection.count.return_value = 10
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            store = VectorStore(
                persist_directory=str(tmp_path / "test_db"),
                collection_name="test_books"
            )
            
            assert store.collection_name == "test_books"
            assert store.persist_directory.exists()
            mock_client.assert_called_once()
            mock_collection.count.assert_called_once()
    
    def test_add_book_chunks(self, vector_store):
        """Test adding book chunks to the store."""
        book_id = "test-book"
        chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
        embeddings = np.random.randn(3, 384)
        metadata_list = [
            {"chapter_id": 0, "chapter_title": "Chapter 1"},
            {"chapter_id": 0, "chapter_title": "Chapter 1"},
            {"chapter_id": 1, "chapter_title": "Chapter 2"}
        ]
        
        vector_store.add_book_chunks(book_id, chunks, embeddings, metadata_list)
        
        # Verify remove_book was called first
        vector_store.collection.get.assert_called_with(where={"book_id": book_id})
        
        # Verify add was called with correct parameters
        vector_store.collection.add.assert_called_once()
        add_call = vector_store.collection.add.call_args
        
        assert len(add_call[1]['ids']) == 3
        assert add_call[1]['ids'][0] == "test-book_chunk_0"
        assert add_call[1]['documents'] == chunks
        assert len(add_call[1]['embeddings']) == 3
        assert all(metadata["book_id"] == book_id for metadata in add_call[1]['metadatas'])
    
    def test_add_book_chunks_validation(self, vector_store):
        """Test validation in add_book_chunks."""
        with pytest.raises(ValueError):
            vector_store.add_book_chunks(
                "test-book",
                ["chunk1", "chunk2"],  # 2 chunks
                np.zeros((3, 384)),    # 3 embeddings - mismatch
                [{"meta": 1}, {"meta": 2}]
            )
    
    def test_remove_book(self, vector_store):
        """Test removing a book from the store."""
        book_id = "test-book"
        
        # Mock existing chunks
        vector_store.collection.get.return_value = {
            'ids': ['test-book_chunk_0', 'test-book_chunk_1', 'test-book_chunk_2']
        }
        
        vector_store.remove_book(book_id)
        
        vector_store.collection.get.assert_called_with(where={"book_id": book_id})
        vector_store.collection.delete.assert_called_with(
            ids=['test-book_chunk_0', 'test-book_chunk_1', 'test-book_chunk_2']
        )
    
    def test_search_with_results(self, vector_store):
        """Test searching with results."""
        query_embedding = np.random.randn(384)
        
        # Mock search results
        vector_store.collection.query.return_value = {
            'ids': [['id1', 'id2']],
            'documents': [['Document 1 content', 'Document 2 content']],
            'metadatas': [[
                {'book_id': 'book1', 'chapter_title': 'Chapter 1'},
                {'book_id': 'book2', 'chapter_title': 'Chapter 2'}
            ]],
            'distances': [[0.5, 0.8]]  # L2 distances
        }
        
        results = vector_store.search(query_embedding, n_results=5, similarity_threshold=0.5)
        
        assert len(results) == 2
        assert results[0]['content'] == 'Document 1 content'
        assert results[0]['similarity'] > results[1]['similarity']  # First result more similar
        assert results[0]['rank'] == 1
        assert results[1]['rank'] == 2
    
    def test_search_with_book_filter(self, vector_store):
        """Test searching with book filter."""
        query_embedding = np.random.randn(384)
        book_filter = ['book1', 'book2']
        
        vector_store.search(query_embedding, n_results=5, book_filter=book_filter)
        
        query_call = vector_store.collection.query.call_args
        assert query_call[1]['where'] == {"book_id": {"$in": book_filter}}
    
    def test_search_with_similarity_threshold(self, vector_store):
        """Test searching with similarity threshold filtering."""
        query_embedding = np.random.randn(384)
        
        # Mock results with varying distances
        vector_store.collection.query.return_value = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [['Doc 1', 'Doc 2', 'Doc 3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.5, 2.0]]  # Corresponds to similarities ~0.91, 0.67, 0.33
        }
        
        results = vector_store.search(query_embedding, n_results=5, similarity_threshold=0.6)
        
        # Only first two results should pass threshold
        assert len(results) == 2
        assert all(result['similarity'] >= 0.6 for result in results)
    
    def test_get_book_chunks(self, vector_store):
        """Test getting all chunks for a book."""
        book_id = "test-book"
        
        vector_store.collection.get.return_value = {
            'documents': ['Chunk 1', 'Chunk 2'],
            'metadatas': [
                {'chapter_id': 0, 'chapter_title': 'Chapter 1'},
                {'chapter_id': 1, 'chapter_title': 'Chapter 2'}
            ]
        }
        
        chunks = vector_store.get_book_chunks(book_id)
        
        assert len(chunks) == 2
        assert chunks[0]['content'] == 'Chunk 1'
        assert chunks[0]['metadata']['chapter_id'] == 0
        vector_store.collection.get.assert_called_with(
            where={"book_id": book_id},
            include=["documents", "metadatas"]
        )
    
    def test_list_books(self, vector_store):
        """Test listing all books."""
        vector_store.collection.get.return_value = {
            'metadatas': [
                {'book_id': 'book1'},
                {'book_id': 'book2'},
                {'book_id': 'book1'},  # Duplicate
                {'book_id': 'book3'}
            ]
        }
        
        books = vector_store.list_books()
        
        assert books == ['book1', 'book2', 'book3']  # Sorted and unique
    
    def test_get_stats(self, vector_store):
        """Test getting store statistics."""
        vector_store.collection.count.return_value = 100
        vector_store.collection.get.return_value = {
            'metadatas': [
                {'book_id': 'book1'},
                {'book_id': 'book2'}
            ]
        }
        
        stats = vector_store.get_stats()
        
        assert stats['total_chunks'] == 100
        assert stats['total_books'] == 2
        assert stats['collection_name'] == 'test_collection'
        assert 'persist_directory' in stats
    
    def test_clear(self, vector_store, mock_chromadb_client):
        """Test clearing the vector store."""
        vector_store.clear()
        
        mock_chromadb_client.delete_collection.assert_called_with(name='test_collection')
        # Should recreate collection
        assert mock_chromadb_client.get_or_create_collection.call_count == 2  # Once in init, once in clear


class TestBookSearchEngine:
    """Test suite for the BookSearchEngine class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=VectorStore)
        store.search.return_value = []
        store.get_book_chunks.return_value = []
        store.list_books.return_value = []
        return store
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """Create a mock embedding generator."""
        generator = Mock()
        generator.embed_text.return_value = np.random.randn(384)
        return generator
    
    @pytest.fixture
    def search_engine(self, mock_vector_store, mock_embedding_generator):
        """Create BookSearchEngine with mocks."""
        return BookSearchEngine(mock_vector_store, mock_embedding_generator)
    
    def test_search_books(self, search_engine, mock_vector_store, mock_embedding_generator):
        """Test searching books with natural language query."""
        query = "machine learning algorithms"
        
        # Mock vector store results
        mock_vector_store.search.return_value = [
            {
                'content': 'Content about ML algorithms',
                'similarity': 0.85,
                'rank': 1,
                'metadata': {
                    'book_id': 'ml-book',
                    'book_title': 'ML Fundamentals',
                    'author': 'John Doe',
                    'chapter_title': 'Chapter 3',
                    'chapter_id': 2
                }
            }
        ]
        
        results = search_engine.search_books(query, max_results=5)
        
        assert len(results) == 1
        assert results[0]['book_title'] == 'ML Fundamentals'
        assert results[0]['similarity'] == 0.85
        assert results[0]['word_count'] == 4
        
        mock_embedding_generator.embed_text.assert_called_once_with(query)
        mock_vector_store.search.assert_called_once()
    
    def test_get_book_content_full_book(self, search_engine, mock_vector_store):
        """Test getting full book content."""
        book_id = "test-book"
        
        mock_vector_store.get_book_chunks.return_value = [
            {
                'content': 'Chapter 1 content',
                'metadata': {
                    'book_title': 'Test Book',
                    'author': 'Test Author',
                    'chapter_id': 0,
                    'chapter_title': 'Chapter 1'
                }
            },
            {
                'content': 'Chapter 2 content',
                'metadata': {
                    'book_title': 'Test Book',
                    'author': 'Test Author',
                    'chapter_id': 1,
                    'chapter_title': 'Chapter 2'
                }
            }
        ]
        
        result = search_engine.get_book_content(book_id)
        
        assert result['book_id'] == book_id
        assert result['book_title'] == 'Test Book'
        assert result['author'] == 'Test Author'
        assert 'Chapter 1 content' in result['content']
        assert 'Chapter 2 content' in result['content']
        assert result['chunk_count'] == 2
    
    def test_get_book_content_specific_chapter(self, search_engine, mock_vector_store):
        """Test getting specific chapter content."""
        book_id = "test-book"
        chapter_id = 1
        
        mock_vector_store.get_book_chunks.return_value = [
            {
                'content': 'Chapter 1 content',
                'metadata': {'chapter_id': 0, 'chapter_title': 'Chapter 1'}
            },
            {
                'content': 'Chapter 2 content',
                'metadata': {
                    'chapter_id': 1,
                    'chapter_title': 'Chapter 2',
                    'book_title': 'Test Book',
                    'author': 'Test Author'
                }
            }
        ]
        
        result = search_engine.get_book_content(book_id, chapter_id)
        
        assert result['chapter_id'] == 1
        assert result['chapter_title'] == 'Chapter 2'
        assert 'Chapter 2 content' in result['content']
        assert 'Chapter 1 content' not in result['content']
    
    def test_get_book_content_not_found(self, search_engine, mock_vector_store):
        """Test getting content for non-existent book."""
        mock_vector_store.get_book_chunks.return_value = []
        
        result = search_engine.get_book_content("non-existent")
        
        assert 'error' in result
        assert 'No content found' in result['error']
    
    def test_find_related_content(self, search_engine, mock_vector_store):
        """Test finding related content."""
        content = "Neural networks and deep learning"
        
        mock_vector_store.list_books.return_value = ['book1', 'book2', 'book3']
        
        # Mock search to be called by search_books
        search_engine.search_books = Mock(return_value=[
            {'book_id': 'book2', 'content': 'Related content'}
        ])
        
        results = search_engine.find_related_content(content, max_results=3, exclude_book='book1')
        
        assert len(results) == 1
        search_engine.search_books.assert_called_with(
            query=content,
            max_results=3,
            book_filter=['book2', 'book3'],
            similarity_threshold=0.6
        )
    
    def test_error_handling_in_search(self, search_engine, mock_embedding_generator):
        """Test error handling in search_books."""
        mock_embedding_generator.embed_text.side_effect = Exception("Embedding error")
        
        with pytest.raises(Exception) as exc_info:
            search_engine.search_books("test query")
        assert "Embedding error" in str(exc_info.value)