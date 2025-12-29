"""
Document parser adapter - interfaces with external parsing API.
Handles PDF, DOCX, images with OCR, etc.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from src.models.schemas import ParsedContent, SourceType, AttachmentType
from src.models.config import ParserConfig

logger = logging.getLogger(__name__)


class DocumentParserError(Exception):
    """Base exception for document parsing errors."""
    pass


class DocumentParser:
    """
    Adapter for external document parsing API.
    Handles retries, timeouts, and error recovery.
    """
    
    def __init__(self, config: ParserConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def parse_attachment(
        self,
        attachment_path: str,
        attachment_type: AttachmentType,
        attachment_name: str
    ) -> List[ParsedContent]:
        """
        Parse a single attachment using external API.
        
        Args:
            attachment_path: Path to the attachment file
            attachment_type: Type of attachment
            attachment_name: Name of the attachment
        
        Returns:
            List of ParsedContent objects
        
        Raises:
            DocumentParserError: If parsing fails after retries
        """
        if attachment_type == AttachmentType.OTHER:
            logger.info(f"Skipping unsupported attachment: {attachment_name}")
            return []
        
        try:
            return await self._parse_with_retry(
                attachment_path,
                attachment_type,
                attachment_name
            )
        except Exception as e:
            logger.error(f"Failed to parse {attachment_name}: {e}")
            # Return empty rather than failing the entire pipeline
            return [ParsedContent(
                source_type=SourceType.ATTACHMENT,
                attachment_name=attachment_name,
                content=f"[PARSE ERROR: {str(e)}]",
                metadata={'error': str(e), 'type': attachment_type.value}
            )]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _parse_with_retry(
        self,
        attachment_path: str,
        attachment_type: AttachmentType,
        attachment_name: str
    ) -> List[ParsedContent]:
        """Parse with automatic retry logic."""
        
        # Read file
        with open(attachment_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare request
        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f"Bearer {self.config.api_key}"
        
        # Simulate external API call
        # In production, this would call Azure Form Recognizer, PyMuPDF API, etc.
        parsed = await self._call_parser_api(
            file_content,
            attachment_type,
            attachment_name,
            headers
        )
        
        return parsed
    
    async def _call_parser_api(
        self,
        file_content: bytes,
        attachment_type: AttachmentType,
        attachment_name: str,
        headers: Dict[str, str]
    ) -> List[ParsedContent]:
        """
        Call external parser API.
        
        THIS IS A MOCK IMPLEMENTATION - Replace with actual API integration.
        """
        
        # Mock response based on file type
        logger.info(f"Parsing {attachment_name} ({attachment_type.value})")
        
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        # Generate mock parsed content
        if attachment_type == AttachmentType.PDF:
            return self._mock_pdf_parse(attachment_name, file_content)
        elif attachment_type in [AttachmentType.DOCX, AttachmentType.DOC]:
            return self._mock_docx_parse(attachment_name, file_content)
        elif attachment_type == AttachmentType.IMAGE:
            return self._mock_ocr_parse(attachment_name, file_content)
        else:
            return []
    
    def _mock_pdf_parse(self, name: str, content: bytes) -> List[ParsedContent]:
        """Mock PDF parsing (replace with real API)."""
        # Simulate multi-page PDF
        return [
            ParsedContent(
                source_type=SourceType.ATTACHMENT,
                attachment_name=name,
                page=1,
                content=f"[Mock PDF content from {name}, page 1]\nThis is sample text extracted from the first page.",
                line_map=[(1, "This is sample text extracted from the first page.")]
            ),
            ParsedContent(
                source_type=SourceType.ATTACHMENT,
                attachment_name=name,
                page=2,
                content=f"[Mock PDF content from {name}, page 2]\nAdditional content from page 2.",
                line_map=[(1, "Additional content from page 2.")]
            )
        ]
    
    def _mock_docx_parse(self, name: str, content: bytes) -> List[ParsedContent]:
        """Mock DOCX parsing (replace with real API)."""
        return [
            ParsedContent(
                source_type=SourceType.ATTACHMENT,
                attachment_name=name,
                content=f"[Mock DOCX content from {name}]\nDocument text extracted from Word file.",
                line_map=[(1, "Document text extracted from Word file.")]
            )
        ]
    
    def _mock_ocr_parse(self, name: str, content: bytes) -> List[ParsedContent]:
        """Mock OCR parsing (replace with real API)."""
        return [
            ParsedContent(
                source_type=SourceType.ATTACHMENT,
                attachment_name=name,
                content=f"[Mock OCR content from {name}]\nText extracted from image via OCR.",
                line_map=[(1, "Text extracted from image via OCR.")]
            )
        ]
    
    async def parse_batch(
        self,
        attachments: List[Dict[str, Any]]
    ) -> List[ParsedContent]:
        """
        Parse multiple attachments in parallel.
        
        Args:
            attachments: List of attachment metadata dicts
        
        Returns:
            Combined list of ParsedContent from all attachments
        """
        tasks = [
            self.parse_attachment(
                att['path'],
                att['type'],
                att['filename']
            )
            for att in attachments
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_parsed = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch parsing error: {result}")
            elif isinstance(result, list):
                all_parsed.extend(result)
        
        return all_parsed
