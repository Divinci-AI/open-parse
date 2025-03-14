from __future__ import annotations
from typing import List, Union, Tuple, Dict, Optional, Literal
from pathlib import Path
from datetime import date
import logging

from markitdown import MarkItDown
from openparse.schemas import Node, FileMetadata, TextElement, Bbox, NodeVariant

class DocumentParser:
    """Parser using Microsoft's MarkItDown for multiple file formats."""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.txt', '.json', '.xml', '.zip'}
    
    def __init__(self, 
                 use_ocr: bool = False, 
                 llm_client: Optional[object] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_tokens: bool = True):
        """
        Initialize the MarkItDown document parser.
        
        Args:
            use_ocr: Whether to use OCR for document parsing
            llm_client: Optional LLM client for enhanced parsing
            chunk_size: Maximum size of text chunks (in characters or tokens)
            chunk_overlap: Number of characters or tokens to overlap between chunks
            use_tokens: If True, measures length in tokens; if False, uses characters
        """
        self.parser = MarkItDown(llm_client=llm_client) if llm_client else MarkItDown()
        self.use_ocr = use_ocr
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_tokens = use_tokens

    def split_text_with_overlap(self, text: str) -> List[str]:
        """
        Splits text into chunks based on paragraphs, respecting max length and overlap.
        
        Args:
            text: The input text to split
            
        Returns:
            List of text chunks
        """
        # Normalize newlines for consistent splitting
        normalized_text = text.replace('\r\n', '\n').replace('\n+', '\n\n').strip()
        
        # Split into paragraphs based on double newlines
        paragraphs = normalized_text.split('\n\n')
        
        chunks = []
        current_chunk = ''
        
        # Helper function to measure length (characters or tokens)
        def get_length(s: str) -> int:
            if self.use_tokens:
                # Rough token count: split by whitespace and filter out empty strings
                return len([word for word in s.split() if word])
            return len(s)  # Character count

        # Helper function to get the last N characters or tokens for overlap
        def get_overlap_segment(s: str, size: int) -> str:
            if self.use_tokens:
                words = [word for word in s.split() if word]
                overlap_words = words[-min(size, len(words)):]
                return ' '.join(overlap_words)
            return s[-min(size, len(s)):]

        for paragraph in paragraphs:
            paragraph_length = get_length(paragraph)

            # If paragraph fits in current chunk
            if get_length(current_chunk) + paragraph_length <= self.chunk_size:
                current_chunk += ('\n\n' if current_chunk else '') + paragraph
            else:
                # If current chunk isn't empty, append it and start a new one with overlap
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = get_overlap_segment(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + '\n\n' + paragraph
                else:
                    # If paragraph alone exceeds max_length, split it further
                    remaining = paragraph
                    while get_length(remaining) > self.chunk_size:
                        if self.use_tokens:
                            # Find approximate token boundary
                            words = [word for word in remaining.split() if word]
                            token_count = 0
                            char_count = 0
                            for i, word in enumerate(words):
                                token_count += 1
                                char_count += len(word) + (1 if i > 0 else 0)  # Add space
                                if token_count >= self.chunk_size:
                                    split_point = char_count
                                    break
                            else:
                                split_point = len(remaining)
                        else:
                            # Find last space before max_length for clean character split
                            split_point = remaining.rfind(' ', 0, self.chunk_size)
                            if split_point == -1:
                                split_point = self.chunk_size

                        chunk = remaining[:split_point].strip()
                        chunks.append(chunk)
                        overlap_text = get_overlap_segment(chunk, self.chunk_overlap)
                        remaining = overlap_text + ' ' + remaining[split_point:].strip()
                    current_chunk = remaining

        # Append the final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def parse_batch(self, files: List[Path], batch_size: int = 1) -> List[Tuple[List[Node], FileMetadata]]:
        """Process multiple files in batches."""
        results = []
        
        for batch in range(0, len(files), batch_size):
            batch_files = files[batch:batch + batch_size]
            for file in batch_files:
                try:
                    result = self.parse(file)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Failed to parse {file}: {e}")
        
        return results
    
    def _get_metadata(self, result, file_path: Path) -> Dict:
        """Extract metadata from MarkItDown result."""
        stats = file_path.stat()
        return {
            "creation_date": None,
            "last_modified_date": date.fromtimestamp(stats.st_mtime),
            "last_accessed_date": date.fromtimestamp(stats.st_atime),
            "file_size": stats.st_size,
            "file_type": file_path.suffix.lower(),
            "is_zip": file_path.suffix.lower() == '.zip'
        }

    def _text_to_nodes(self, text: str, start_page: int = 1) -> List[Node]:
        """Convert text content to nodes with RAG-based chunking."""
        nodes = []
        if text and len(text.strip()) > 0:
            # Apply RAG-based chunking
            chunks = self.split_text_with_overlap(text)
            
            self.logger.debug(f"Split text into {len(chunks)} chunks using RAG-based chunking")
            
            for i, chunk in enumerate(chunks, start_page):
                if chunk.strip():
                    element = TextElement(
                        text=chunk.strip(),
                        lines=(),
                        bbox=Bbox(
                            page=i,
                            page_height=1000,
                            page_width=1000,
                            x0=0, y0=0,
                            x1=1000, y1=1000
                        ),
                        variant=NodeVariant.TEXT
                    )
                    nodes.append(Node(
                        elements=(element,),
                        bbox=element.bbox
                    ))
        return nodes

    def parse(self, file: Union[str, Path]) -> Tuple[List[Node], FileMetadata]:
        """Parse document into nodes using MarkItDown."""
        file_path = Path(file)
        file_extension = file_path.suffix.lower()
        
        # Handle case where file extension might be empty
        if not file_extension and isinstance(file, str):
            # Try to extract extension from filename
            filename = file.split('/')[-1]
            if '.' in filename:
                file_extension = '.' + filename.split('.')[-1].lower()
        
        if not file_extension:
            self.logger.warning(f"No file extension detected for {file_path}, attempting to infer from content")
            # Default to PDF if we can't determine the extension
            file_extension = '.pdf'
        
        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"❌ Unsupported file format: {file_extension}")
            
        try:
            # Pass file extension for ZIP handling
            result = self.parser.convert_local(
                str(file_path), 
                file_extension=file_extension
            )
            metadata = self._get_metadata(result, file_path)
            metadata['file_type'] = file_extension  # Ensure file_type is set correctly
            
            text = result.text_content
            self.logger.debug(f"📑 Extracted text content: {text[:100]}...")
            
            nodes = self._text_to_nodes(text)
            self.logger.debug(f"🔢 Created {len(nodes)} nodes from document")
            
            # Add page count to metadata
            metadata['page_count'] = len(nodes) if nodes else 1
            
            return nodes, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error details: {str(e)}", exc_info=True)
            raise ValueError(f"❌ Failed to parse {file_path}: {str(e)}")
