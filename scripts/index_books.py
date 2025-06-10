#!/usr/bin/env python3
"""Script to index EPUB books into the vector store."""

import sys
import logging
from pathlib import Path
import json
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.parsers.epub_parser import EPUBParser
from src.embeddings.embeddings import EmbeddingGenerator
from src.search.vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def index_epub_file(epub_path: Path, parser: EPUBParser, embedding_generator: EmbeddingGenerator, 
                   vector_store: VectorStore) -> bool:
    """
    Index a single EPUB file.
    
    Args:
        epub_path: Path to EPUB file
        parser: EPUB parser instance
        embedding_generator: Embedding generator instance
        vector_store: Vector store instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {epub_path.name}")
        
        # Parse EPUB
        metadata, chapters = parser.parse_epub(str(epub_path))
        
        if not chapters:
            logger.warning(f"No chapters found in {epub_path.name}")
            return False
        
        # Save processed book data
        processed_file = parser.save_processed_book(
            metadata, chapters, config.books['processed_dir']
        )
        logger.info(f"Saved processed data to: {processed_file}")
        
        # Prepare chunks and metadata for embedding
        all_chunks = []
        all_metadata = []
        
        chunk_size = config.vector_store['chunk_size']
        chunk_overlap = config.vector_store['chunk_overlap']
        
        for chapter in chapters:
            # Chunk the chapter content
            chunks = embedding_generator.chunk_text(
                chapter.content, chunk_size, chunk_overlap
            )
            
            # Create metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = {
                    "book_id": metadata.id,
                    "book_title": metadata.title,
                    "author": metadata.author,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "chunk_index": chunk_idx,
                    "chunk_word_count": len(chunk.split())
                }
                
                all_chunks.append(chunk)
                all_metadata.append(chunk_metadata)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(chapters)} chapters")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedding_generator.embed_texts(all_chunks, show_progress=True)
        
        # Add to vector store
        logger.info("Adding to vector store...")
        vector_store.add_book_chunks(
            book_id=metadata.id,
            chunks=all_chunks,
            embeddings=embeddings,
            metadata_list=all_metadata
        )
        
        logger.info(f"Successfully indexed {epub_path.name}")
        logger.info(f"  - Book ID: {metadata.id}")
        logger.info(f"  - Chapters: {len(chapters)}")
        logger.info(f"  - Chunks: {len(all_chunks)}")
        logger.info(f"  - Total words: {sum(ch.word_count for ch in chapters)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error indexing {epub_path.name}: {e}")
        return False

def main():
    """Main indexing function."""
    # Initialize components
    logger.info("Initializing components...")
    
    try:
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(
            model_name=config.embeddings['model'],
            device=config.embeddings['device'],
            cache_dir=config.embeddings['cache_dir']
        )
        
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory=config.books['index_dir'],
            collection_name=config.vector_store['collection_name']
        )
        
        # Initialize parser
        parser = EPUBParser()
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Get list of EPUB files
    epub_dir = Path(config.books['data_dir'])
    if not epub_dir.exists():
        logger.error(f"EPUB directory not found: {epub_dir}")
        sys.exit(1)
    
    epub_files = list(epub_dir.glob("*.epub"))
    if not epub_files:
        logger.warning(f"No EPUB files found in {epub_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(epub_files)} EPUB files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for epub_file in epub_files:
        logger.info(f"\n{'='*50}")
        if index_epub_file(epub_file, parser, embedding_generator, vector_store):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("INDEXING COMPLETE")
    logger.info(f"Successfully processed: {successful} books")
    logger.info(f"Failed to process: {failed} books")
    
    # Print vector store stats
    stats = vector_store.get_stats()
    logger.info(f"\nVector Store Statistics:")
    logger.info(f"  - Total books: {stats['total_books']}")
    logger.info(f"  - Total chunks: {stats['total_chunks']}")
    logger.info(f"  - Available books: {', '.join(stats['books'])}")

if __name__ == "__main__":
    main()