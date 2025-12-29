"""
Email parsing adapter for .eml and .msg files.
"""

import email
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from extract_msg import Message as MSGMessage

from src.models.schemas import ParsedContent, SourceType, AttachmentType

logger = logging.getLogger(__name__)


class EmailParser:
    """Parse .eml and .msg email files."""
    
    def __init__(self):
        self.supported_extensions = {'.eml', '.msg'}
    
    async def parse(self, email_path: str) -> Dict[str, Any]:
        """
        Parse an email file and extract headers, body, and attachments.
        
        Args:
            email_path: Path to the email file
            
        Returns:
            Dict containing:
                - headers: Email headers
                - body: Email body text
                - attachments: List of attachment metadata
        
        Raises:
            ValueError: If file format is not supported
            IOError: If file cannot be read
        """
        path = Path(email_path)
        
        if path.suffix not in self.supported_extensions:
            raise ValueError(
                f"Unsupported email format: {path.suffix}. "
                f"Supported formats: {self.supported_extensions}"
            )
        
        try:
            if path.suffix == '.eml':
                return await self._parse_eml(path)
            elif path.suffix == '.msg':
                return await self._parse_msg(path)
        except Exception as e:
            logger.error(f"Failed to parse email {email_path}: {e}")
            raise
    
    async def _parse_eml(self, path: Path) -> Dict[str, Any]:
        """Parse .eml file."""
        with open(path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        # Extract headers
        headers = {
            'from': msg.get('From', ''),
            'to': msg.get('To', ''),
            'subject': msg.get('Subject', ''),
            'date': msg.get('Date', ''),
            'message_id': msg.get('Message-ID', ''),
        }
        
        # Extract body
        body = self._extract_body(msg)
        
        # Extract attachments
        attachments = self._extract_attachments_eml(msg, path.parent)
        
        return {
            'headers': headers,
            'body': body,
            'attachments': attachments
        }
    
    async def _parse_msg(self, path: Path) -> Dict[str, Any]:
        """Parse .msg file."""
        msg = MSGMessage(str(path))
        
        # Extract headers
        headers = {
            'from': msg.sender or '',
            'to': msg.to or '',
            'subject': msg.subject or '',
            'date': msg.date or '',
            'message_id': '',
        }
        
        # Extract body
        body = msg.body or ''
        
        # Extract attachments
        attachments = self._extract_attachments_msg(msg, path.parent)
        
        msg.close()
        
        return {
            'headers': headers,
            'body': body,
            'attachments': attachments
        }
    
    def _extract_body(self, msg: email.message.EmailMessage) -> str:
        """Extract email body, preferring plain text over HTML."""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                
                # Get text content
                if content_type == 'text/plain':
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break  # Prefer plain text
                    except Exception as e:
                        logger.warning(f"Failed to decode plain text: {e}")
                
                elif content_type == 'text/html' and not body:
                    try:
                        # Fallback to HTML if no plain text
                        html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        # Simple HTML stripping (in production, use BeautifulSoup)
                        body = self._strip_html_simple(html)
                    except Exception as e:
                        logger.warning(f"Failed to decode HTML: {e}")
        else:
            # Single part message
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to decode message body: {e}")
        
        return body.strip()
    
    def _strip_html_simple(self, html: str) -> str:
        """Simple HTML tag stripping (use BeautifulSoup in production)."""
        import re
        # Remove script and style elements
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_attachments_eml(self, msg: email.message.EmailMessage, base_path: Path) -> List[Dict[str, Any]]:
        """Extract attachment metadata from .eml message."""
        attachments = []
        attachment_num = 0
        
        for part in msg.walk():
            content_disposition = str(part.get('Content-Disposition', ''))
            
            if 'attachment' in content_disposition:
                filename = part.get_filename()
                if not filename:
                    # Generate filename
                    ext = part.get_content_type().split('/')[-1]
                    filename = f"attachment_{attachment_num}.{ext}"
                    attachment_num += 1
                
                # Save attachment temporarily
                attachment_path = base_path / "temp_attachments" / filename
                attachment_path.parent.mkdir(exist_ok=True)
                
                try:
                    with open(attachment_path, 'wb') as f:
                        f.write(part.get_payload(decode=True))
                    
                    attachments.append({
                        'filename': filename,
                        'path': str(attachment_path),
                        'content_type': part.get_content_type(),
                        'size': attachment_path.stat().st_size,
                        'type': self._classify_attachment(filename)
                    })
                except Exception as e:
                    logger.error(f"Failed to save attachment {filename}: {e}")
        
        return attachments
    
    def _extract_attachments_msg(self, msg: MSGMessage, base_path: Path) -> List[Dict[str, Any]]:
        """Extract attachment metadata from .msg message."""
        attachments = []
        
        for attachment in msg.attachments:
            filename = attachment.longFilename or attachment.shortFilename or f"attachment_{len(attachments)}"
            
            # Save attachment temporarily
            attachment_path = base_path / "temp_attachments" / filename
            attachment_path.parent.mkdir(exist_ok=True)
            
            try:
                with open(attachment_path, 'wb') as f:
                    f.write(attachment.data)
                
                attachments.append({
                    'filename': filename,
                    'path': str(attachment_path),
                    'content_type': attachment.mimetype or 'application/octet-stream',
                    'size': len(attachment.data),
                    'type': self._classify_attachment(filename)
                })
            except Exception as e:
                logger.error(f"Failed to save attachment {filename}: {e}")
        
        return attachments
    
    def _classify_attachment(self, filename: str) -> AttachmentType:
        """Classify attachment by file extension."""
        ext = Path(filename).suffix.lower()
        
        if ext == '.pdf':
            return AttachmentType.PDF
        elif ext in ['.docx']:
            return AttachmentType.DOCX
        elif ext in ['.doc']:
            return AttachmentType.DOC
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            return AttachmentType.IMAGE
        else:
            return AttachmentType.OTHER
