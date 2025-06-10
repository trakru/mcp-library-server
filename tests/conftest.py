import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        'embeddings': {
            'model_name': 'all-MiniLM-L6-v2',
            'device': 'cpu',
            'batch_size': 32,
            'max_length': 512
        },
        'generation': {
            'model': 'qwen2.5:7b',
            'temperature': 0.7,
            'max_tokens': 2000,
            'timeout': 30
        },
        'search': {
            'top_k': 5,
            'similarity_threshold': 0.7
        },
        'vector_db': {
            'persist_directory': './test_vector_db',
            'collection_name': 'test_books'
        },
        'data': {
            'epub_directory': './test_data/epub'
        },
        'logging': {
            'level': 'INFO',
            'file': 'test_logs/mcp_server.log'
        }
    }

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files"""
    return tmp_path

@pytest.fixture
def mock_epub_file(temp_dir):
    """Create a mock EPUB file structure"""
    epub_dir = temp_dir / "test_book"
    epub_dir.mkdir()
    
    # Create minimal EPUB structure
    meta_inf = epub_dir / "META-INF"
    meta_inf.mkdir()
    
    container_xml = meta_inf / "container.xml"
    container_xml.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>''')
    
    content_opf = epub_dir / "content.opf"
    content_opf.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
    <metadata>
        <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Test Book</dc:title>
        <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">Test Author</dc:creator>
    </metadata>
    <manifest>
        <item id="chapter1" href="chapter1.html" media-type="application/xhtml+xml"/>
    </manifest>
    <spine>
        <itemref idref="chapter1"/>
    </spine>
</package>''')
    
    chapter1 = epub_dir / "chapter1.html"
    chapter1.write_text('''<html>
<head><title>Chapter 1</title></head>
<body>
<h1>Chapter 1: Introduction</h1>
<p>This is test content for chapter 1.</p>
</body>
</html>''')
    
    return epub_dir

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    client = Mock()
    client.is_healthy.return_value = True
    client.generate.return_value = "Generated content based on the input."
    return client

@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for testing"""
    generator = Mock()
    generator.generate_embedding.return_value = [0.1] * 384  # Mock embedding vector
    generator.chunk_text.return_value = ["chunk1", "chunk2"]
    return generator

@pytest.fixture
def mock_chromadb_client():
    """Mock ChromaDB client for testing"""
    collection = MagicMock()
    collection.count.return_value = 0
    collection.add = MagicMock()
    collection.query = MagicMock(return_value={
        'ids': [['doc1', 'doc2']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[
            {'book_title': 'Test Book', 'chapter': 'Chapter 1'},
            {'book_title': 'Test Book', 'chapter': 'Chapter 2'}
        ]],
        'documents': [['Document 1 content', 'Document 2 content']]
    })
    
    client = Mock()
    client.get_or_create_collection.return_value = collection
    
    return client

@pytest.fixture(autouse=True)
def cleanup_test_dirs(request):
    """Clean up test directories after tests"""
    yield
    # Cleanup code here if needed
    test_dirs = ['./test_vector_db', './test_logs', './test_data']
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)