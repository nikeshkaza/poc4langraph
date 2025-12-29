"""
Pydantic models for the email extraction system.
Defines all data structures with validation and type safety.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Type of content source."""
    EMAIL_BODY = "email_body"
    ATTACHMENT = "attachment"


class AttachmentType(str, Enum):
    """Supported attachment types."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    IMAGE = "image"
    OTHER = "other"


class ParsedContent(BaseModel):
    """Normalized content structure from any source."""
    source_type: SourceType
    attachment_name: Optional[str] = None
    page: Optional[int] = None
    content: str
    line_map: Optional[List[tuple[int, str]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Individual text chunk with provenance."""
    chunk_id: str
    source_type: SourceType
    attachment_name: Optional[str] = None
    page: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Citation metadata for extracted fields."""
    source: SourceType
    attachment_name: Optional[str] = None
    chunk_id: str
    page: Optional[int] = None
    line_numbers: Optional[List[int]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_pass: Literal["initial", "retry"] = "initial"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationRule(BaseModel):
    """Field validation rule."""
    rule_type: Literal["regex", "range", "enum", "custom"]
    value: Any
    error_message: Optional[str] = None


class FieldMetadata(BaseModel):
    """Metadata for a single extraction field."""
    name: str
    description: str
    category: str
    field_type: Literal["string", "number", "date", "boolean", "list"] = "string"
    required: bool = False
    priority: int = Field(default=1, ge=1, le=5)
    validation_rules: List[ValidationRule] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)


class ExtractedField(BaseModel):
    """Extracted field with value and metadata."""
    name: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    citation: Citation
    validated: bool = True
    validation_errors: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """State maintained by each category agent."""
    agent_id: str
    category: str
    assigned_fields: List[FieldMetadata]
    extracted_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    chunks_scanned: int = 0
    retry_count: int = 0
    errors: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ExtractionMetrics(BaseModel):
    """Metrics for the extraction process."""
    total_fields: int
    extracted_fields: int
    fill_rate: float = Field(ge=0.0, le=1.0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    retry_lift: float = 0.0  # % improvement from retries
    total_chunks: int
    chunks_scanned: int
    processing_time_seconds: float
    agent_times: Dict[str, float] = Field(default_factory=dict)
    errors_count: int = 0


class ErrorRecord(BaseModel):
    """Structured error information."""
    error_type: str
    error_message: str
    component: str  # Which node/agent failed
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recoverable: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Final extraction result."""
    document_id: str
    email_subject: Optional[str] = None
    fields: Dict[str, ExtractedField]
    metrics: ExtractionMetrics
    errors: List[ErrorRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GraphState(BaseModel):
    """Complete state passed through the LangGraph workflow."""
    
    # Input
    document_id: str
    email_path: str
    
    # Email content
    email_headers: Dict[str, str] = Field(default_factory=dict)
    email_body: str = ""
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Parsed content
    parsed_contents: List[ParsedContent] = Field(default_factory=list)
    
    # Chunks
    chunks: List[Chunk] = Field(default_factory=list)
    
    # Field metadata
    field_metadata: List[FieldMetadata] = Field(default_factory=list)
    
    # Agent states
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    
    # Extracted fields (aggregated)
    extracted_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 2
    retry_fields: List[str] = Field(default_factory=list)
    
    # Metrics & errors
    metrics: Optional[ExtractionMetrics] = None
    errors: List[ErrorRecord] = Field(default_factory=list)
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True


class LLMRequest(BaseModel):
    """Request to LLM for field extraction."""
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 2000
    response_format: Literal["json", "text"] = "json"


class LLMResponse(BaseModel):
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
