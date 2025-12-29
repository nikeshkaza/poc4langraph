"""
Test suite for the email extraction system.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime

from src.models.schemas import (
    FieldMetadata, ParsedContent, Chunk,
    SourceType, ExtractedField, Citation
)
from src.models.config import ExtractionConfig, ChunkingConfig
from src.utils.chunking import TextChunker
from src.parsers.email_parser import EmailParser


class TestFieldMetadata:
    """Test field metadata model."""
    
    def test_field_creation(self):
        field = FieldMetadata(
            name="company_name",
            description="Legal company name",
            category="company_info",
            field_type="string",
            required=True,
            priority=5
        )
        
        assert field.name == "company_name"
        assert field.required is True
        assert field.priority == 5
    
    def test_field_validation(self):
        with pytest.raises(ValueError):
            FieldMetadata(
                name="test",
                description="Test",
                category="test",
                priority=10  # Invalid: should be 1-5
            )


class TestChunking:
    """Test text chunking functionality."""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=50
        )
        return TextChunker(config)
    
    def test_chunk_small_text(self, chunker):
        content = ParsedContent(
            source_type=SourceType.EMAIL_BODY,
            content="Short text"
        )
        
        chunks = chunker.chunk_content(content)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"
    
    def test_chunk_long_text(self, chunker):
        long_text = "word " * 100  # 500 chars
        content = ParsedContent(
            source_type=SourceType.EMAIL_BODY,
            content=long_text
        )
        
        chunks = chunker.chunk_content(content)
        assert len(chunks) > 1
        
        # Verify overlap
        if len(chunks) > 1:
            assert any(
                chunks[i].content[-20:] in chunks[i+1].content
                for i in range(len(chunks)-1)
            )
    
    def test_chunk_preserves_metadata(self, chunker):
        content = ParsedContent(
            source_type=SourceType.ATTACHMENT,
            attachment_name="test.pdf",
            page=1,
            content="Test content"
        )
        
        chunks = chunker.chunk_content(content)
        assert chunks[0].source_type == SourceType.ATTACHMENT
        assert chunks[0].attachment_name == "test.pdf"
        assert chunks[0].page == 1


@pytest.mark.asyncio
class TestEmailParser:
    """Test email parsing."""
    
    @pytest.fixture
    def parser(self):
        return EmailParser()
    
    async def test_unsupported_format(self, parser):
        with pytest.raises(ValueError):
            await parser.parse("test.txt")
    
    def test_classify_attachment(self, parser):
        assert parser._classify_attachment("doc.pdf") == "pdf"
        assert parser._classify_attachment("doc.docx") == "docx"
        assert parser._classify_attachment("img.png") == "image"
        assert parser._classify_attachment("data.xyz") == "other"


class TestExtractedField:
    """Test extracted field model."""
    
    def test_field_with_citation(self):
        citation = Citation(
            source=SourceType.EMAIL_BODY,
            chunk_id="test-chunk",
            confidence=0.95,
            extraction_pass="initial"
        )
        
        field = ExtractedField(
            name="test_field",
            value="test value",
            confidence=0.95,
            citation=citation
        )
        
        assert field.confidence == 0.95
        assert field.citation.extraction_pass == "initial"
    
    def test_confidence_validation(self):
        citation = Citation(
            source=SourceType.EMAIL_BODY,
            chunk_id="test",
            confidence=0.5
        )
        
        with pytest.raises(ValueError):
            ExtractedField(
                name="test",
                value="test",
                confidence=1.5,  # Invalid: > 1.0
                citation=citation
            )


class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_from_dict(self):
        config_dict = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4"
            },
            "parser": {
                "endpoint": "https://api.example.com"
            },
            "chunking": {
                "chunk_size": 1000
            },
            "retry": {
                "max_retries": 2
            }
        }
        
        config = ExtractionConfig(**config_dict)
        assert config.llm.model == "gpt-4"
        assert config.retry.max_retries == 2


class TestWorkflowIntegration:
    """Integration tests for the full workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_extraction(self):
        """
        This would be a full end-to-end test.
        In production, you'd create a test email file and validate the output.
        """
        # Mock implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
