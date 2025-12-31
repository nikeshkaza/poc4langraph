"""
Document parser adapter - interfaces with external parsing API.
Handles PDF, DOCX, images with OCR, etc.
Uses PyMuPDF for PDFs and Azure Document Intelligence for other formats.
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
import os

# PDF parsing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyMuPDF not installed. PDF parsing will be limited.")

# Azure Document Intelligence
try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Azure SDK not installed. Azure Document Intelligence will not be available.")

from src.models.schemas import ParsedContent, SourceType, AttachmentType
from src.models.config import ParserConfig

logger = logging.getLogger(__name__)


class DocumentParserError(Exception):
    """Base exception for document parsing errors."""
    pass


class PDFParsingError(DocumentParserError):
    """Error parsing PDF document."""
    pass


class AzureParsingError(DocumentParserError):
    """Error with Azure Document Intelligence."""
    pass


class DocumentParser:
    """
    Adapter for external document parsing API.
    Handles retries, timeouts, and error recovery.
    """
    
    def __init__(self, config: ParserConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize Azure Document Intelligence client if available
        self.azure_client = None
        if AZURE_AVAILABLE:
            azure_endpoint = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
            azure_key = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
            
            if azure_endpoint and azure_key:
                try:
                    self.azure_client = DocumentAnalysisClient(
                        endpoint=azure_endpoint,
                        credential=AzureKeyCredential(azure_key)
                    )
                    logger.info("Azure Document Intelligence client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Azure client: {e}")
            else:
                logger.info("Azure credentials not found in environment. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        # Check PyMuPDF availability
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available. Install with: pip install PyMuPDF")
    
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
        
        # Route to appropriate parser based on file type
        if attachment_type == AttachmentType.PDF:
            return await self._parse_pdf_pymupdf(attachment_path, attachment_name)
        elif attachment_type in [AttachmentType.DOCX, AttachmentType.DOC]:
            return await self._parse_with_azure(attachment_path, attachment_name, 'document')
        elif attachment_type == AttachmentType.IMAGE:
            return await self._parse_with_azure(attachment_path, attachment_name, 'image')
        else:
            logger.warning(f"Unsupported attachment type: {attachment_type}")
            return []
    
    
    async def _parse_pdf_pymupdf(
        self,
        pdf_path: str,
        attachment_name: str
    ) -> List[ParsedContent]:
        """
        Parse PDF using PyMuPDF (fitz).
        
        Args:
            pdf_path: Path to PDF file
            attachment_name: Name of the attachment
            
        Returns:
            List of ParsedContent, one per page
        """
        if not PYMUPDF_AVAILABLE:
            raise PDFParsingError("PyMuPDF not installed. Install with: pip install PyMuPDF")
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            parsed_contents = []
            
            logger.info(f"Parsing PDF: {attachment_name} ({doc.page_count} pages)")
            
            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text("text")
                
                # Build line map
                line_map = []
                lines = text.split('\n')
                for line_no, line_text in enumerate(lines, 1):
                    if line_text.strip():
                        line_map.append((line_no, line_text))
                
                # Create ParsedContent
                parsed_content = ParsedContent(
                    source_type=SourceType.ATTACHMENT,
                    attachment_name=attachment_name,
                    page=page_num + 1,  # 1-indexed
                    content=text,
                    line_map=line_map,
                    metadata={
                        'page_count': doc.page_count,
                        'page_width': page.rect.width,
                        'page_height': page.rect.height,
                        'parser': 'pymupdf'
                    }
                )
                parsed_contents.append(parsed_content)
                
                logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")
            
            doc.close()
            
            logger.info(f"Successfully parsed PDF: {attachment_name} - {doc.page_count} pages, {sum(len(p.content) for p in parsed_contents)} total characters")
            
            return parsed_contents
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {attachment_name}: {e}")
            raise PDFParsingError(f"PyMuPDF parsing failed: {str(e)}")
    
    async def _parse_with_azure(
        self,
        file_path: str,
        attachment_name: str,
        doc_type: str = 'document'
    ) -> List[ParsedContent]:
        """
        Parse document using Azure Document Intelligence.
        
        Args:
            file_path: Path to the file
            attachment_name: Name of the attachment
            doc_type: Type of document ('document' for DOCX/DOC, 'image' for images)
            
        Returns:
            List of ParsedContent
        """
        if not self.azure_client:
            raise AzureParsingError(
                "Azure Document Intelligence not initialized. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables"
            )
        
        try:
            logger.info(f"Parsing {doc_type} with Azure: {attachment_name}")
            
            # Read file
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Start analysis - using prebuilt-read model for general text extraction
            poller = self.azure_client.begin_analyze_document(
                model_id="prebuilt-read",
                document=file_content
            )
            
            # Wait for completion
            result = poller.result()
            
            parsed_contents = []
            page_count = len(result.pages)
            
            logger.info(f"Azure analysis complete: {page_count} pages detected")
            
            # Extract content per page
            for page_num, page in enumerate(result.pages, 1):
                # Collect all text from lines on this page
                page_text = []
                line_map = []
                
                for line in page.lines:
                    page_text.append(line.content)
                    line_map.append((len(line_map) + 1, line.content))
                
                full_text = '\n'.join(page_text)
                
                # Create ParsedContent
                parsed_content = ParsedContent(
                    source_type=SourceType.ATTACHMENT,
                    attachment_name=attachment_name,
                    page=page_num,
                    content=full_text,
                    line_map=line_map,
                    metadata={
                        'page_count': page_count,
                        'page_width': page.width,
                        'page_height': page.height,
                        'page_angle': page.angle,
                        'unit': page.unit,
                        'parser': 'azure_document_intelligence',
                        'model_id': 'prebuilt-read'
                    }
                )
                parsed_contents.append(parsed_content)
                
                logger.debug(f"Extracted {len(full_text)} characters from page {page_num}")
            
            logger.info(f"Successfully parsed with Azure: {attachment_name} - {page_count} pages, {sum(len(p.content) for p in parsed_contents)} total characters")
            
            return parsed_contents
            
        except Exception as e:
            logger.error(f"Azure Document Intelligence failed for {attachment_name}: {e}")
            raise AzureParsingError(f"Azure parsing failed: {str(e)}")
    
    def _extract_document_metadata(
        self,
        parsed_contents: List[ParsedContent]
    ) -> Dict[str, Any]:
        """
        Extract summary metadata from parsed contents.
        
        Args:
            parsed_contents: List of parsed content
            
        Returns:
            Dictionary with metadata
        """
        if not parsed_contents:
            return {
                'page_count': 0,
                'total_characters': 0,
                'total_lines': 0
            }
        
        # Calculate totals
        page_count = len(parsed_contents)
        total_characters = sum(len(p.content) for p in parsed_contents)
        total_lines = sum(len(p.line_map) if p.line_map else 0 for p in parsed_contents)
        
        # Get parser used
        parser = parsed_contents[0].metadata.get('parser', 'unknown')
        
        return {
            'page_count': page_count,
            'total_characters': total_characters,
            'total_lines': total_lines,
            'parser_used': parser,
            'attachment_name': parsed_contents[0].attachment_name
        }
    
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
