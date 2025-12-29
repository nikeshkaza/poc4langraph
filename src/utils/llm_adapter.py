"""
LLM adapter for field extraction.
Interfaces with GPT-4, Claude, or Azure OpenAI.
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import (
    LLMRequest, LLMResponse, FieldMetadata,
    Chunk, ExtractedField, Citation
)
from src.models.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Adapter for LLM providers (OpenAI, Anthropic, Azure).
    Handles extraction prompting and response parsing.
    """
    
    def __init__(self, config: LLMConfig):
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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True
    )
    async def extract_field(
        self,
        field: FieldMetadata,
        chunks: List[Chunk],
        is_retry: bool = False
    ) -> Optional[ExtractedField]:
        """
        Extract a single field from chunks using LLM.
        
        Args:
            field: Field metadata
            chunks: List of text chunks to search
            is_retry: Whether this is a retry attempt
        
        Returns:
            ExtractedField if found, None otherwise
        """
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(field, chunks, is_retry)
        
        # Call LLM
        try:
            response = await self._call_llm(prompt)
            
            # Parse response
            extracted = self._parse_extraction_response(response, field, chunks)
            
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to extract field {field.name}: {e}")
            return None
    
    def _build_extraction_prompt(
        self,
        field: FieldMetadata,
        chunks: List[Chunk],
        is_retry: bool
    ) -> LLMRequest:
        """Build structured prompt for field extraction."""
        
        system_prompt = """You are a precise field extraction assistant.
Your task is to extract specific fields from document chunks.

Rules:
1. Extract ONLY the requested field value
2. Return confidence score (0.0-1.0) based on certainty
3. Cite the EXACT chunk_id where you found the value
4. If not found, return null
5. Be conservative - only extract if reasonably confident

Return JSON format:
{
    "found": true/false,
    "value": "extracted value" or null,
    "confidence": 0.0-1.0,
    "chunk_id": "id" or null,
    "reasoning": "brief explanation"
}
"""
        
        # Build user prompt
        chunk_texts = []
        for i, chunk in enumerate(chunks[:20]):  # Limit to first 20 chunks
            chunk_info = f"[CHUNK {i+1} - ID: {chunk.chunk_id}]"
            if chunk.attachment_name:
                chunk_info += f" [Source: {chunk.attachment_name}"
                if chunk.page:
                    chunk_info += f", Page {chunk.page}"
                chunk_info += "]"
            chunk_texts.append(f"{chunk_info}\n{chunk.content}\n")
        
        user_prompt = f"""Field to extract: {field.name}
Description: {field.description}
Type: {field.field_type}
"""
        
        if field.examples:
            user_prompt += f"\nExamples: {', '.join(field.examples)}"
        
        if is_retry and field.synonyms:
            user_prompt += f"\nAlso look for synonyms: {', '.join(field.synonyms)}"
        
        user_prompt += "\n\nDocument chunks:\n\n" + "\n".join(chunk_texts)
        user_prompt += "\n\nExtract the field value from above chunks."
        
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0 if not is_retry else 0.3,  # Slightly higher temp on retry
            response_format="json"
        )
    
    async def _call_llm(self, request: LLMRequest) -> LLMResponse:
        """
        Call the LLM API.
        
        THIS IS A MOCK IMPLEMENTATION - Replace with actual API calls.
        """
        
        # Simulate API call
        await asyncio.sleep(0.2)
        
        # Mock response
        mock_content = json.dumps({
            "found": True,
            "value": "Sample Extracted Value",
            "confidence": 0.85,
            "chunk_id": "mock-chunk-id",
            "reasoning": "Found clear match in document"
        })
        
        return LLMResponse(
            content=mock_content,
            model=self.config.model,
            tokens_used=150,
            finish_reason="stop"
        )
    
    def _parse_extraction_response(
        self,
        response: LLMResponse,
        field: FieldMetadata,
        chunks: List[Chunk]
    ) -> Optional[ExtractedField]:
        """Parse LLM response into ExtractedField."""
        
        try:
            data = json.loads(response.content)
            
            if not data.get('found') or data.get('value') is None:
                return None
            
            # Find the source chunk
            chunk_id = data.get('chunk_id')
            source_chunk = next(
                (c for c in chunks if c.chunk_id == chunk_id),
                chunks[0] if chunks else None
            )
            
            if not source_chunk:
                logger.warning(f"Chunk {chunk_id} not found for field {field.name}")
                return None
            
            # Create citation
            citation = Citation(
                source=source_chunk.source_type,
                attachment_name=source_chunk.attachment_name,
                chunk_id=source_chunk.chunk_id,
                page=source_chunk.page,
                line_numbers=[source_chunk.start_line] if source_chunk.start_line else None,
                confidence=float(data.get('confidence', 0.5)),
                extraction_pass="initial"
            )
            
            # Create extracted field
            return ExtractedField(
                name=field.name,
                value=data['value'],
                confidence=citation.confidence,
                citation=citation
            )
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            return None
    
    async def extract_fields_batch(
        self,
        fields: List[FieldMetadata],
        chunks: List[Chunk],
        is_retry: bool = False
    ) -> Dict[str, Optional[ExtractedField]]:
        """
        Extract multiple fields in parallel.
        
        Args:
            fields: List of fields to extract
            chunks: Chunks to search
            is_retry: Whether this is a retry pass
        
        Returns:
            Dict mapping field names to extracted values
        """
        tasks = [
            self.extract_field(field, chunks, is_retry)
            for field in fields
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        extracted = {}
        for field, result in zip(fields, results):
            if isinstance(result, Exception):
                logger.error(f"Error extracting {field.name}: {result}")
                extracted[field.name] = None
            else:
                extracted[field.name] = result
        
        return extracted
