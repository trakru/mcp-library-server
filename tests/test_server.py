import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import asyncio

from server import (
    search_books,
    get_chapter_content,
    generate_section,
    explain_concept,
    cite_sources,
    list_available_books,
    get_current_config
)


class TestMCPServerTools:
    """Test suite for the MCP server tool functions."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup mocks for all server components."""
        self.mock_search_engine = Mock()
        self.mock_doc_generator = Mock()
        self.mock_vector_store = Mock()
        self.mock_config = Mock()
        
        # Patch global objects in server module
        self.patches = [
            patch('server.search_engine', self.mock_search_engine),
            patch('server.doc_generator', self.mock_doc_generator),
            patch('server.vector_store', self.mock_vector_store),
            patch('server.config', self.mock_config)
        ]
        
        for p in self.patches:
            p.start()
        
        # Setup config mock
        self.mock_config.search = {'similarity_threshold': 0.7}
        
        yield
        
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_search_books_success(self):
        """Test successful book search."""
        # Mock search results
        mock_results = [
            {
                'similarity': 0.85,
                'book_title': 'ML Fundamentals',
                'author': 'John Doe',
                'chapter_title': 'Chapter 1',
                'content': 'Machine learning is a subset of artificial intelligence...',
                'book_id': 'ml-fundamentals',
                'chapter_id': 0
            },
            {
                'similarity': 0.72,
                'book_title': 'Deep Learning',
                'author': 'Jane Smith',
                'chapter_title': 'Neural Networks',
                'content': 'Neural networks are computational models inspired by biological neurons...',
                'book_id': 'deep-learning',
                'chapter_id': 2
            }
        ]
        
        self.mock_search_engine.search_books.return_value = mock_results
        
        result = search_books("machine learning algorithms", max_results=5)
        result_data = json.loads(result)
        
        assert result_data['query'] == "machine learning algorithms"
        assert result_data['total_results'] == 2
        assert len(result_data['results']) == 2
        
        # Check first result formatting
        first_result = result_data['results'][0]
        assert first_result['rank'] == 1
        assert first_result['similarity'] == 0.85
        assert first_result['book'] == "ML Fundamentals by John Doe"
        assert first_result['chapter'] == "Chapter 1"
        assert 'book_id' in first_result
        assert 'full_content' in first_result
        
        # Verify search_engine was called correctly
        self.mock_search_engine.search_books.assert_called_once_with(
            query="machine learning algorithms",
            max_results=5,
            book_filter=None,
            similarity_threshold=0.7
        )
    
    def test_search_books_with_filter(self):
        """Test book search with book filter."""
        self.mock_search_engine.search_books.return_value = []
        
        search_books("neural networks", max_results=3, book_filter=["book1", "book2"])
        
        self.mock_search_engine.search_books.assert_called_once_with(
            query="neural networks",
            max_results=3,
            book_filter=["book1", "book2"],
            similarity_threshold=0.7
        )
    
    def test_search_books_error(self):
        """Test search books with error."""
        self.mock_search_engine.search_books.side_effect = Exception("Search failed")
        
        result = search_books("test query")
        result_data = json.loads(result)
        
        assert 'error' in result_data
        assert "Search failed" in result_data['error']
    
    def test_get_chapter_content_success(self):
        """Test successful chapter content retrieval."""
        mock_content = {
            'book_title': 'Test Book',
            'author': 'Test Author',
            'content': 'Chapter content here...',
            'chapter_title': 'Test Chapter',
            'chunk_count': 3,
            'word_count': 150
        }
        
        self.mock_search_engine.get_book_content.return_value = mock_content
        
        result = get_chapter_content("test-book", chapter_id=1, format="markdown")
        result_data = json.loads(result)
        
        assert result_data['book_title'] == 'Test Book'
        assert result_data['author'] == 'Test Author'
        assert '# Test Book' in result_data['content']
        assert '## Chapter 1:' in result_data['content']
        assert result_data['metadata']['chunk_count'] == 3
        
        self.mock_search_engine.get_book_content.assert_called_once_with("test-book", 1)
    
    def test_get_chapter_content_plain_format(self):
        """Test chapter content retrieval in plain format."""
        mock_content = {
            'book_title': 'Test Book',
            'author': 'Test Author',
            'content': 'Plain text content...',
            'chunk_count': 1,
            'word_count': 50
        }
        
        self.mock_search_engine.get_book_content.return_value = mock_content
        
        result = get_chapter_content("test-book", format="plain")
        result_data = json.loads(result)
        
        assert result_data['content'] == 'Plain text content...'
        assert '# Test Book' not in result_data['content']
    
    def test_get_chapter_content_not_found(self):
        """Test chapter content retrieval for non-existent content."""
        self.mock_search_engine.get_book_content.return_value = {
            "error": "No content found for book test-book"
        }
        
        result = get_chapter_content("test-book")
        result_data = json.loads(result)
        
        assert 'error' in result_data
        assert "No content found" in result_data['error']
    
    def test_generate_section_success(self):
        """Test successful section generation."""
        # Mock search results
        mock_search_results = [
            {
                'content': 'Source content 1...',
                'book_title': 'Book 1',
                'author': 'Author 1',
                'chapter_title': 'Chapter A',
                'similarity': 0.9
            },
            {
                'content': 'Source content 2...',
                'book_title': 'Book 2',
                'author': 'Author 2',
                'chapter_title': 'Chapter B',
                'similarity': 0.8
            }
        ]
        
        self.mock_search_engine.search_books.return_value = mock_search_results
        self.mock_doc_generator.generate_section.return_value = "Generated documentation section..."
        
        result = generate_section("neural networks", max_sources=3, style="tutorial", length="brief")
        result_data = json.loads(result)
        
        assert result_data['topic'] == "neural networks"
        assert result_data['style'] == "tutorial"
        assert result_data['length'] == "brief"
        assert result_data['generated_content'] == "Generated documentation section..."
        assert len(result_data['sources_used']) == 2
        
        # Check source formatting
        first_source = result_data['sources_used'][0]
        assert first_source['source_number'] == 1
        assert first_source['book'] == "Book 1 by Author 1"
        assert first_source['similarity'] == 0.9
        
        # Verify calls
        self.mock_search_engine.search_books.assert_called_once_with(
            query="neural networks",
            max_results=3,
            similarity_threshold=0.7
        )
        
        self.mock_doc_generator.generate_section.assert_called_once_with(
            topic="neural networks",
            source_content=['Source content 1...', 'Source content 2...'],
            style="tutorial",
            max_length="brief"
        )
    
    def test_generate_section_no_sources(self):
        """Test section generation with no sources found."""
        self.mock_search_engine.search_books.return_value = []
        
        result = generate_section("nonexistent topic")
        result_data = json.loads(result)
        
        assert 'error' in result_data
        assert "No relevant content found" in result_data['error']
    
    def test_explain_concept_success(self):
        """Test successful concept explanation."""
        mock_search_results = [
            {
                'content': 'Gradient descent explanation...',
                'book_title': 'Optimization Book',
                'author': 'Math Expert',
                'chapter_title': 'Optimization',
                'similarity': 0.95
            }
        ]
        
        self.mock_search_engine.search_books.return_value = mock_search_results
        self.mock_doc_generator.explain_concept.return_value = "Detailed explanation of gradient descent..."
        
        result = explain_concept("gradient descent", include_examples=True, max_sources=2)
        result_data = json.loads(result)
        
        assert result_data['concept'] == "gradient descent"
        assert result_data['explanation'] == "Detailed explanation of gradient descent..."
        assert result_data['include_examples'] is True
        assert len(result_data['sources_used']) == 1
        
        # Verify search query includes "definition explanation"
        search_call = self.mock_search_engine.search_books.call_args[1]
        assert "gradient descent definition explanation" in search_call['query']
        
        self.mock_doc_generator.explain_concept.assert_called_once_with(
            concept="gradient descent",
            source_content=['Gradient descent explanation...'],
            include_examples=True
        )
    
    def test_cite_sources_success(self):
        """Test successful source citation."""
        mock_chunks = [
            {
                'metadata': {
                    'book_title': 'Machine Learning Yearning',
                    'author': 'Andrew Ng'
                }
            }
        ]
        
        self.mock_vector_store.get_book_chunks.return_value = mock_chunks
        
        result = cite_sources("ml-yearning", format="APA")
        result_data = json.loads(result)
        
        assert result_data['book_id'] == "ml-yearning"
        assert result_data['format'] == "APA"
        assert "Andrew Ng" in result_data['citation']
        assert "Machine Learning Yearning" in result_data['citation']
        assert result_data['metadata']['title'] == "Machine Learning Yearning"
        assert result_data['metadata']['author'] == "Andrew Ng"
    
    def test_cite_sources_book_not_found(self):
        """Test citation for non-existent book."""
        self.mock_vector_store.get_book_chunks.return_value = []
        
        result = cite_sources("nonexistent-book")
        result_data = json.loads(result)
        
        assert 'error' in result_data
        assert "Book nonexistent-book not found" in result_data['error']
    
    def test_list_available_books_success(self):
        """Test listing available books."""
        mock_stats = {
            'books': ['book1', 'book2'],
            'total_chunks': 100
        }
        
        mock_chunks_book1 = [
            {'metadata': {'book_title': 'Book 1', 'author': 'Author 1'}, 'content': 'content 1'},
            {'metadata': {'book_title': 'Book 1', 'author': 'Author 1'}, 'content': 'content 2'}
        ]
        
        mock_chunks_book2 = [
            {'metadata': {'book_title': 'Book 2', 'author': 'Author 2'}, 'content': 'longer content here'}
        ]
        
        self.mock_vector_store.get_stats.return_value = mock_stats
        self.mock_vector_store.get_book_chunks.side_effect = [mock_chunks_book1, mock_chunks_book2]
        
        result = list_available_books()
        result_data = json.loads(result)
        
        assert result_data['total_books'] == 2
        assert result_data['total_chunks'] == 100
        assert len(result_data['books']) == 2
        
        # Check first book
        book1 = result_data['books'][0]
        assert book1['book_id'] == 'book1'
        assert book1['title'] == 'Book 1'
        assert book1['author'] == 'Author 1'
        assert book1['total_chunks'] == 2
        assert book1['total_words'] == 4  # "content" + "1" + "content" + "2"
    
    def test_get_current_config(self):
        """Test getting current configuration."""
        # Setup mock config attributes
        self.mock_config.embeddings = {'model': 'all-MiniLM-L6-v2', 'device': 'cpu'}
        self.mock_config.generation = {'model': 'qwen2.5:7b', 'base_url': 'http://localhost:11434'}
        self.mock_config.search = {'max_results': 5, 'similarity_threshold': 0.7}
        self.mock_config.vector_store = {'collection_name': 'books', 'chunk_size': 512}
        
        result = get_current_config()
        result_data = json.loads(result)
        
        assert result_data['embeddings']['model'] == 'all-MiniLM-L6-v2'
        assert result_data['generation']['model'] == 'qwen2.5:7b'
        assert result_data['search']['similarity_threshold'] == 0.7
        assert result_data['vector_store']['collection_name'] == 'books'
    
    def test_error_handling_in_tools(self):
        """Test error handling across different tools."""
        # Test error in generate_section
        self.mock_search_engine.search_books.side_effect = Exception("Search error")
        
        result = generate_section("test topic")
        result_data = json.loads(result)
        assert 'error' in result_data
        
        # Reset mock
        self.mock_search_engine.search_books.side_effect = None
        
        # Test error in explain_concept
        self.mock_doc_generator.explain_concept.side_effect = Exception("Generation error")
        self.mock_search_engine.search_books.return_value = [{'content': 'test'}]
        
        result = explain_concept("test concept")
        result_data = json.loads(result)
        assert 'error' in result_data
        
        # Test error in list_available_books
        self.mock_vector_store.get_stats.side_effect = Exception("Vector store error")
        
        result = list_available_books()
        result_data = json.loads(result)
        assert 'error' in result_data