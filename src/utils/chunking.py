"""
Text chunking utilities for semantic segmentation.
"""

from typing import List
import uuid
import logging

from src.models.schemas import ParsedContent, Chunk, SourceType
from src.models.config import ChunkingConfig

logger = logging.getLogger(__name__)


class TextChunker:
    """Semantic text chunking with provenance preservation."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def chunk_content(self, parsed_content: ParsedContent) -> List[Chunk]:
        """
        Chunk a single ParsedContent into smaller pieces.
        
        Args:
            parsed_content: The content to chunk
        
        Returns:
            List of Chunks with preserved metadata
        """
        text = parsed_content.content
        
        if len(text) < self.config.min_chunk_size:
            # Text is too small, create single chunk
            return [self._create_chunk(parsed_content, text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.config.chunk_size, len(text))
            
            # Try to break at separator if not at end
            if end < len(text):
                # Look for separator within overlap range
                separator_pos = text.rfind(
                    self.config.separator,
                    end - self.config.chunk_overlap,
                    end
                )
                if separator_pos != -1:
                    end = separator_pos + len(self.config.separator)
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = self._create_chunk(parsed_content, chunk_text, start, end)
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.config.chunk_overlap
            
            # Prevent infinite loop
            if start <= chunks[-1].metadata.get('start_pos', 0) if chunks else False:
                break
        
        logger.info(
            f"Created {len(chunks)} chunks from {parsed_content.source_type.value} "
            f"({parsed_content.attachment_name or 'email body'})"
        )
        
        return chunks
    
    def _create_chunk(
        self,
        parsed_content: ParsedContent,
        chunk_text: str,
        start_pos: int,
        end_pos: int
    ) -> Chunk:
        """Create a Chunk object with provenance."""
        
        # Calculate line numbers if line_map exists
        start_line, end_line = None, None
        if parsed_content.line_map and self.config.preserve_line_numbers:
            start_line, end_line = self._find_line_numbers(
                parsed_content.line_map,
                start_pos,
                end_pos
            )
        
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            source_type=parsed_content.source_type,
            attachment_name=parsed_content.attachment_name,
            page=parsed_content.page,
            start_line=start_line,
            end_line=end_line,
            content=chunk_text,
            metadata={
                'start_pos': start_pos,
                'end_pos': end_pos,
                'length': len(chunk_text),
                **parsed_content.metadata
            }
        )
    
    def _find_line_numbers(
        self,
        line_map: List[tuple[int, str]],
        start_pos: int,
        end_pos: int
    ) -> tuple[int, int]:
        """Find line numbers corresponding to character positions."""
        start_line = None
        end_line = None
        
        current_pos = 0
        for line_no, line_text in line_map:
            line_start = current_pos
            line_end = current_pos + len(line_text)
            
            if start_line is None and line_end > start_pos:
                start_line = line_no
            
            if line_end >= end_pos:
                end_line = line_no
                break
            
            current_pos = line_end + 1  # +1 for newline
        
        return start_line or 1, end_line or len(line_map)
    
    def chunk_all(self, parsed_contents: List[ParsedContent]) -> List[Chunk]:
        """
        Chunk all parsed content.
        
        Args:
            parsed_contents: List of ParsedContent objects
        
        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        
        for content in parsed_contents:
            try:
                chunks = self.chunk_content(content)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(
                    f"Failed to chunk content from "
                    f"{content.attachment_name or 'email body'}: {e}"
                )
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
