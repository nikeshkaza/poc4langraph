"""
Category extraction agent - processes a specific field category.
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging

from src.models.schemas import (
    AgentState, FieldMetadata, Chunk,
    ExtractedField, SourceType
)
from src.utils.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


class CategoryAgent:
    """
    Agent responsible for extracting fields in a specific category.
    
    Each agent:
    - Processes its assigned fields independently
    - Maintains its own state
    - Tracks confidence and provenance
    - Handles errors gracefully
    """
    
    def __init__(
        self,
        agent_id: str,
        category: str,
        fields: List[FieldMetadata],
        llm_adapter: LLMAdapter
    ):
        self.agent_id = agent_id
        self.category = category
        self.fields = fields
        self.llm_adapter = llm_adapter
        
        self.state = AgentState(
            agent_id=agent_id,
            category=category,
            assigned_fields=fields,
            missing_fields=[f.name for f in fields]
        )
    
    async def extract(self, chunks: List[Chunk], is_retry: bool = False) -> AgentState:
        """
        Execute extraction for all assigned fields.
        
        Args:
            chunks: All available chunks
            is_retry: Whether this is a retry pass
        
        Returns:
            Updated agent state
        """
        self.state.start_time = datetime.utcnow()
        
        try:
            logger.info(
                f"Agent {self.agent_id} starting extraction: "
                f"{len(self.state.missing_fields)} fields, "
                f"{len(chunks)} chunks"
            )
            
            # Extract fields
            if is_retry:
                # On retry, only process missing fields
                fields_to_extract = [
                    f for f in self.fields
                    if f.name in self.state.missing_fields
                ]
            else:
                fields_to_extract = self.fields
            
            # Process fields
            extracted = await self.llm_adapter.extract_fields_batch(
                fields_to_extract,
                chunks,
                is_retry
            )
            
            # Update state
            for field_name, extracted_field in extracted.items():
                self.state.chunks_scanned += len(chunks)
                
                if extracted_field:
                    # Field successfully extracted
                    if is_retry:
                        extracted_field.citation.extraction_pass = "retry"
                    
                    self.state.extracted_fields[field_name] = extracted_field
                    
                    # Remove from missing
                    if field_name in self.state.missing_fields:
                        self.state.missing_fields.remove(field_name)
                    
                    logger.info(
                        f"Agent {self.agent_id} extracted {field_name} "
                        f"(confidence: {extracted_field.confidence:.2f})"
                    )
            
            if is_retry:
                self.state.retry_count += 1
            
        except Exception as e:
            error_msg = f"Agent {self.agent_id} error: {str(e)}"
            logger.error(error_msg)
            self.state.errors.append(error_msg)
        
        finally:
            self.state.end_time = datetime.utcnow()
        
        logger.info(
            f"Agent {self.agent_id} completed: "
            f"{len(self.state.extracted_fields)}/{len(self.fields)} fields extracted"
        )
        
        return self.state
    
    def get_missing_fields(self) -> List[str]:
        """Get list of fields that haven't been extracted yet."""
        return self.state.missing_fields.copy()
    
    def get_extracted_fields(self) -> Dict[str, ExtractedField]:
        """Get all successfully extracted fields."""
        return self.state.extracted_fields.copy()
    
    def get_low_confidence_fields(self, threshold: float = 0.7) -> List[str]:
        """Get fields extracted with low confidence."""
        return [
            name for name, field in self.state.extracted_fields.items()
            if field.confidence < threshold
        ]
    
    def get_processing_time(self) -> float:
        """Get total processing time in seconds."""
        if self.state.start_time and self.state.end_time:
            delta = self.state.end_time - self.state.start_time
            return delta.total_seconds()
        return 0.0


class AgentOrchestrator:
    """Coordinates multiple category agents."""
    
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter
        self.agents: Dict[str, CategoryAgent] = {}
    
    def create_agents(
        self,
        field_metadata: List[FieldMetadata]
    ) -> Dict[str, CategoryAgent]:
        """
        Create category agents based on field metadata.
        
        Args:
            field_metadata: All field definitions
        
        Returns:
            Dict mapping category names to agents
        """
        # Group fields by category
        categories: Dict[str, List[FieldMetadata]] = {}
        for field in field_metadata:
            if field.category not in categories:
                categories[field.category] = []
            categories[field.category].append(field)
        
        # Create agent for each category
        agents = {}
        for i, (category, fields) in enumerate(categories.items()):
            agent_id = f"agent_{i+1}_{category}"
            agent = CategoryAgent(
                agent_id=agent_id,
                category=category,
                fields=fields,
                llm_adapter=self.llm_adapter
            )
            agents[category] = agent
            
            logger.info(
                f"Created agent {agent_id} for category '{category}' "
                f"with {len(fields)} fields"
            )
        
        self.agents = agents
        return agents
    
    async def run_all_agents(
        self,
        chunks: List[Chunk],
        is_retry: bool = False
    ) -> Dict[str, AgentState]:
        """
        Run all agents in parallel.
        
        Args:
            chunks: All available chunks
            is_retry: Whether this is a retry pass
        
        Returns:
            Dict mapping category to agent states
        """
        import asyncio
        
        logger.info(f"Running {len(self.agents)} agents in parallel")
        
        tasks = [
            agent.extract(chunks, is_retry)
            for agent in self.agents.values()
        ]
        
        states = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for (category, agent), state in zip(self.agents.items(), states):
            if isinstance(state, Exception):
                logger.error(f"Agent {category} failed: {state}")
                # Return partial state
                result[category] = agent.state
            else:
                result[category] = state
        
        return result
