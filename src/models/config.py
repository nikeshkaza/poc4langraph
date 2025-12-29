"""
Configuration models for the extraction system.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl
import yaml
from pathlib import Path


class LLMConfig(BaseModel):
    """LLM endpoint configuration."""
    provider: str = "openai"  # openai, anthropic, azure
    api_key: Optional[str] = None
    endpoint: Optional[HttpUrl] = None
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 30
    max_retries: int = 3


class ParserConfig(BaseModel):
    """Document parser API configuration."""
    endpoint: HttpUrl
    api_key: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    supported_types: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "doc", "png", "jpg", "jpeg"]
    )


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    separator: str = "\n\n"
    preserve_line_numbers: bool = True


class RetryConfig(BaseModel):
    """Retry logic configuration."""
    max_retries: int = 2
    retry_missing_only: bool = True
    relaxed_confidence_threshold: float = 0.5  # Lower threshold on retry
    use_synonyms: bool = True
    expand_context: bool = True  # Use surrounding chunks


class ExtractionConfig(BaseModel):
    """Main extraction configuration."""
    
    # System config
    system_name: str = "email-extraction-system"
    version: str = "1.0.0"
    
    # Component configs
    llm: LLMConfig
    parser: ParserConfig
    chunking: ChunkingConfig
    retry: RetryConfig
    
    # Field metadata path
    field_metadata_path: Optional[str] = None
    
    # Agent configuration
    parallel_agents: int = 5
    agent_timeout: int = 300  # seconds
    
    # Confidence thresholds
    min_confidence: float = 0.7
    high_confidence: float = 0.9
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_path: Optional[str] = "logs/extraction.log"
    
    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_validation: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExtractionConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


class FieldCategory(BaseModel):
    """Field category grouping."""
    name: str
    description: str
    fields: List[str]
    priority: int = 1
