import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from ebooklib import epub
import ebooklib

from src.parsers.epub_parser import EPUBParser, BookMetadata, BookChapter


class TestEPUBParser:
    """Test suite for the EPUBParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create EPUBParser instance."""
        return EPUBParser()
    
    @pytest.fixture
    def mock_epub_book(self):
        """Create a mock EPUB book."""
        book = Mock(spec=epub.EpubBook)
        
        # Mock metadata
        book.get_metadata.side_effect = lambda namespace, name: {
            ('DC', 'title'): [["Test Book Title"]],
            ('DC', 'creator'): [["Test Author"]],
            ('DC', 'date'): [["2024-01-01"]],
            ('DC', 'language'): [["en"]],
            ('DC', 'description'): [["Test description"]],
        }.get((namespace, name), [])
        
        # Mock document items
        item1 = Mock()
        item1.get_type.return_value = ebooklib.ITEM_DOCUMENT
        item1.get_content.return_value = b"""
        <html>
            <head><title>Chapter 1</title></head>
            <body>
                <h1>Chapter 1: Introduction</h1>
                <p>This is the first chapter with some content that is long enough to be considered valid.</p>
            </body>
        </html>
        """
        
        item2 = Mock()
        item2.get_type.return_value = ebooklib.ITEM_DOCUMENT
        item2.get_content.return_value = b"""
        <html>
            <body>
                <h2 class="chapter-title">Chapter 2: Main Content</h2>
                <p>This is the second chapter with even more content to ensure it passes the length threshold.</p>
            </body>
        </html>
        """
        
        # Mock item that should be skipped (too short)
        item3 = Mock()
        item3.get_type.return_value = ebooklib.ITEM_DOCUMENT
        item3.get_content.return_value = b"<html><body><p>Too short</p></body></html>"
        
        # Mock non-document item
        item4 = Mock()
        item4.get_type.return_value = ebooklib.ITEM_IMAGE
        
        book.get_items.return_value = [item1, item2, item3, item4]
        
        return book
    
    def test_parse_epub_success(self, parser, mock_epub_book, tmp_path):
        """Test successful EPUB parsing."""
        # Create a dummy EPUB file
        epub_path = tmp_path / "test_book.epub"
        epub_path.touch()
        
        with patch('ebooklib.epub.read_epub', return_value=mock_epub_book):
            metadata, chapters = parser.parse_epub(str(epub_path))
        
        # Check metadata
        assert metadata.title == "Test Book Title"
        assert metadata.author == "Test Author"
        assert metadata.publication_date == "2024-01-01"
        assert metadata.language == "en"
        assert metadata.description == "Test description"
        assert metadata.id == "test_book"
        
        # Check chapters
        assert len(chapters) == 2  # Only 2 chapters should pass the length threshold
        assert chapters[0].title == "Chapter 1: Introduction"
        assert "first chapter" in chapters[0].content
        assert chapters[1].title == "Chapter 2: Main Content"
        assert "second chapter" in chapters[1].content
    
    def test_parse_epub_file_not_found(self, parser):
        """Test parsing non-existent EPUB file."""
        with pytest.raises(FileNotFoundError):
            parser.parse_epub("non_existent.epub")
    
    def test_parse_epub_with_error(self, parser, tmp_path):
        """Test error handling during EPUB parsing."""
        epub_path = tmp_path / "corrupted.epub"
        epub_path.touch()
        
        with patch('ebooklib.epub.read_epub', side_effect=Exception("Corrupt EPUB")):
            with pytest.raises(Exception) as exc_info:
                parser.parse_epub(str(epub_path))
            assert "Corrupt EPUB" in str(exc_info.value)
    
    def test_extract_metadata_with_defaults(self, parser, tmp_path):
        """Test metadata extraction with missing values."""
        book = Mock(spec=epub.EpubBook)
        book.get_metadata.return_value = []  # No metadata
        
        epub_path = tmp_path / "test_book.epub"
        metadata = parser._extract_metadata(book, epub_path)
        
        assert metadata.title == "test_book"  # Falls back to filename
        assert metadata.author == "Unknown Author"
        assert metadata.publication_date == "Unknown"
        assert metadata.language == "en"
        assert metadata.description == ""
    
    def test_generate_book_id(self, parser):
        """Test book ID generation from filename."""
        assert parser._generate_book_id("Machine Learning Basics.epub") == "machine-learning-basics"
        assert parser._generate_book_id("book_with_special@chars!.pdf") == "book_with_special-chars"
        assert parser._generate_book_id("book---with---dashes") == "book-with-dashes"
        assert parser._generate_book_id("") == "unknown-book"
        assert parser._generate_book_id("UPPERCASE.EPUB") == "uppercase"
    
    def test_extract_text_from_html(self, parser):
        """Test HTML text extraction."""
        html = b"""
        <html>
            <head>
                <script>console.log('should be removed');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <h1>Title</h1>
                <p>First paragraph.</p>
                <p>Second   paragraph   with   spaces.</p>
            </body>
        </html>
        """
        
        text = parser._extract_text_from_html(html)
        
        assert "console.log" not in text
        assert "color: red" not in text
        assert "Title" in text
        assert "First paragraph." in text
        assert "Second paragraph with spaces." in text  # Multiple spaces should be normalized
    
    def test_extract_text_from_html_error(self, parser):
        """Test HTML text extraction with invalid HTML."""
        invalid_html = b"Not valid HTML at all <><><"
        
        # Should not raise exception, just return empty or partial text
        text = parser._extract_text_from_html(invalid_html)
        assert isinstance(text, str)
    
    def test_extract_title_from_html(self, parser):
        """Test title extraction from HTML."""
        # Test h1 tag
        html1 = b"<html><body><h1>Chapter Title</h1></body></html>"
        assert parser._extract_title_from_html(html1) == "Chapter Title"
        
        # Test title tag
        html2 = b"<html><head><title>Page Title</title></head></html>"
        assert parser._extract_title_from_html(html2) == "Page Title"
        
        # Test class-based title
        html3 = b'<html><body><div class="chapter-title">Class Title</div></body></html>'
        assert parser._extract_title_from_html(html3) == "Class Title"
        
        # Test no title
        html4 = b"<html><body><p>Just some text</p></body></html>"
        assert parser._extract_title_from_html(html4) is None
        
        # Test overly long title (should be rejected)
        html5 = b"<html><body><h1>" + b"A" * 300 + b"</h1></body></html>"
        assert parser._extract_title_from_html(html5) is None
    
    def test_save_processed_book(self, parser, tmp_path):
        """Test saving processed book data."""
        metadata = BookMetadata(
            id="test-book",
            title="Test Book",
            author="Test Author",
            publication_date="2024",
            language="en",
            description="Test description"
        )
        
        chapters = [
            BookChapter(
                chapter_id=0,
                title="Chapter 1",
                content="Content of chapter 1",
                word_count=4,
                book_id="test-book",
                book_title="Test Book",
                author="Test Author"
            ),
            BookChapter(
                chapter_id=1,
                title="Chapter 2",
                content="Content of chapter 2 with more words",
                word_count=7,
                book_id="test-book",
                book_title="Test Book",
                author="Test Author"
            )
        ]
        
        output_dir = tmp_path / "output"
        saved_path = parser.save_processed_book(metadata, chapters, str(output_dir))
        
        # Check file was created
        assert Path(saved_path).exists()
        assert saved_path == str(output_dir / "test-book.json")
        
        # Check file content
        with open(saved_path, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["id"] == "test-book"
        assert data["metadata"]["title"] == "Test Book"
        assert data["metadata"]["total_chapters"] == 2
        assert data["metadata"]["total_words"] == 11
        
        assert len(data["chapters"]) == 2
        assert data["chapters"][0]["title"] == "Chapter 1"
        assert data["chapters"][1]["word_count"] == 7
    
    def test_book_chapter_dataclass(self):
        """Test BookChapter dataclass."""
        chapter = BookChapter(
            chapter_id=1,
            title="Test Chapter",
            content="Test content",
            word_count=2,
            book_id="test-id",
            book_title="Test Book",
            author="Test Author"
        )
        
        assert chapter.chapter_id == 1
        assert chapter.title == "Test Chapter"
        assert chapter.word_count == 2
    
    def test_book_metadata_dataclass(self):
        """Test BookMetadata dataclass."""
        metadata = BookMetadata(
            id="test-id",
            title="Test Title",
            author="Test Author",
            publication_date="2024",
            language="en",
            description="Test description"
        )
        
        assert metadata.id == "test-id"
        assert metadata.title == "Test Title"
        assert metadata.language == "en"