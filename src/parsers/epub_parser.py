"""EPUB parser for extracting and structuring book content."""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BookChapter:
    """Represents a chapter from a book."""
    chapter_id: int
    title: str
    content: str
    word_count: int
    book_id: str
    book_title: str
    author: str

@dataclass
class BookMetadata:
    """Book metadata."""
    id: str
    title: str
    author: str
    publication_date: str
    language: str
    description: str

class EPUBParser:
    """Parser for EPUB files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_epub(self, epub_path: str) -> tuple[BookMetadata, List[BookChapter]]:
        """
        Parse an EPUB file and extract metadata and chapters.
        
        Args:
            epub_path: Path to the EPUB file
            
        Returns:
            Tuple of (metadata, chapters)
        """
        epub_path = Path(epub_path)
        if not epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")
        
        self.logger.info(f"Parsing EPUB: {epub_path}")
        
        try:
            book = epub.read_epub(str(epub_path))
            metadata = self._extract_metadata(book, epub_path)
            chapters = self._extract_chapters(book, metadata)
            
            self.logger.info(f"Successfully parsed {len(chapters)} chapters from {metadata.title}")
            return metadata, chapters
            
        except Exception as e:
            self.logger.error(f"Error parsing EPUB {epub_path}: {e}")
            raise
    
    def _extract_metadata(self, book: epub.EpubBook, epub_path: Path) -> BookMetadata:
        """Extract metadata from EPUB book."""
        # Generate book ID from filename
        book_id = self._generate_book_id(epub_path.stem)
        
        # Extract metadata with fallbacks
        title = self._get_metadata_value(book, 'DC', 'title') or epub_path.stem
        author = self._get_metadata_value(book, 'DC', 'creator') or "Unknown Author"
        publication_date = self._get_metadata_value(book, 'DC', 'date') or "Unknown"
        language = self._get_metadata_value(book, 'DC', 'language') or "en"
        description = self._get_metadata_value(book, 'DC', 'description') or ""
        
        return BookMetadata(
            id=book_id,
            title=title,
            author=author,
            publication_date=publication_date,
            language=language,
            description=description
        )
    
    def _extract_chapters(self, book: epub.EpubBook, metadata: BookMetadata) -> List[BookChapter]:
        """Extract chapters from EPUB book."""
        chapters = []
        chapter_id = 0
        
        # Get all document items
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = self._extract_text_from_html(item.get_content())
                
                if content and len(content.strip()) > 100:  # Skip very short content
                    # Try to extract title from the HTML
                    title = self._extract_title_from_html(item.get_content()) or f"Chapter {chapter_id + 1}"
                    
                    chapter = BookChapter(
                        chapter_id=chapter_id,
                        title=title,
                        content=content,
                        word_count=len(content.split()),
                        book_id=metadata.id,
                        book_title=metadata.title,
                        author=metadata.author
                    )
                    
                    chapters.append(chapter)
                    chapter_id += 1
        
        return chapters
    
    def _get_metadata_value(self, book: epub.EpubBook, namespace: str, name: str) -> Optional[str]:
        """Safely extract metadata value."""
        try:
            metadata = book.get_metadata(namespace, name)
            if metadata:
                return metadata[0][0]  # Get first value
        except Exception:
            pass
        return None
    
    def _extract_text_from_html(self, html_content: bytes) -> str:
        """Extract clean text from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up line breaks
            text = text.strip()
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Error extracting text from HTML: {e}")
            return ""
    
    def _extract_title_from_html(self, html_content: bytes) -> Optional[str]:
        """Extract title from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for title in various header tags
            for tag in ['h1', 'h2', 'h3', 'title']:
                title_elem = soup.find(tag)
                if title_elem:
                    title = title_elem.get_text().strip()
                    if title and len(title) < 200:  # Reasonable title length
                        return title
            
            # Look for class names that might indicate titles
            for class_name in ['title', 'chapter-title', 'heading']:
                title_elem = soup.find(class_=class_name)
                if title_elem:
                    title = title_elem.get_text().strip()
                    if title and len(title) < 200:
                        return title
                        
        except Exception:
            pass
        
        return None
    
    def _generate_book_id(self, filename: str) -> str:
        """Generate a clean book ID from filename."""
        # Remove common file extensions and clean up
        book_id = filename.lower()
        book_id = re.sub(r'\.(epub|pdf)$', '', book_id)
        book_id = re.sub(r'[^\w\-_]', '-', book_id)
        book_id = re.sub(r'-+', '-', book_id)
        book_id = book_id.strip('-')
        
        return book_id or "unknown-book"
    
    def save_processed_book(self, metadata: BookMetadata, chapters: List[BookChapter], output_dir: str) -> str:
        """
        Save processed book data to JSON file.
        
        Args:
            metadata: Book metadata
            chapters: List of chapters
            output_dir: Directory to save the processed data
            
        Returns:
            Path to the saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{metadata.id}.json"
        
        data = {
            "metadata": {
                "id": metadata.id,
                "title": metadata.title,
                "author": metadata.author,
                "publication_date": metadata.publication_date,
                "language": metadata.language,
                "description": metadata.description,
                "total_chapters": len(chapters),
                "total_words": sum(ch.word_count for ch in chapters)
            },
            "chapters": [
                {
                    "chapter_id": ch.chapter_id,
                    "title": ch.title,
                    "content": ch.content,
                    "word_count": ch.word_count
                }
                for ch in chapters
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved processed book to: {output_file}")
        return str(output_file)