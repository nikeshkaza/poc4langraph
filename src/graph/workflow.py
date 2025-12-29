"""
LangGraph workflow definition for email extraction.

This module defines the complete extraction pipeline as a LangGraph StateGraph.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.schemas import (
    GraphState, ParsedContent, SourceType,
    ExtractedField, ExtractionMetrics, ErrorRecord
)
from src.models.config import ExtractionConfig
from src.parsers.email_parser import EmailParser
from src.parsers.document_parser import DocumentParser
from src.utils.chunking import TextChunker
from src.utils.llm_adapter import LLMAdapter
from src.agents.category_agent import AgentOrchestrator

logger = logging.getLogger(__name__)


class ExtractionWorkflow:
    """
    LangGraph-based extraction workflow.
    
    Workflow Steps:
    1. preprocess: Parse email and attachments
    2. parse_documents: Extract text from attachments
    3. chunk_content: Create searchable chunks
    4. parallel_extract: Run category agents in parallel
    5. aggregate: Combine results from all agents
    6. check_completion: Decide if retry is needed
    7. retry_extraction: Re-extract missing fields (conditional)
    8. finalize: Compute metrics and assemble output
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph StateGraph."""
        
        # Create workflow graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("parse_documents", self._parse_documents_node)
        workflow.add_node("chunk_content", self._chunk_content_node)
        workflow.add_node("parallel_extract", self._parallel_extract_node)
        workflow.add_node("aggregate", self._aggregate_node)
        workflow.add_node("retry_extraction", self._retry_extraction_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "parse_documents")
        workflow.add_edge("parse_documents", "chunk_content")
        workflow.add_edge("chunk_content", "parallel_extract")
        workflow.add_edge("parallel_extract", "aggregate")
        
        # Conditional edge: retry or finalize?
        workflow.add_conditional_edges(
            "aggregate",
            self._should_retry,
            {
                "retry": "retry_extraction",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("retry_extraction", "aggregate")
        workflow.add_edge("finalize", END)
        
        # Compile graph
        self.graph = workflow.compile()
        
        logger.info("LangGraph workflow compiled successfully")
    
    async def _preprocess_node(self, state: GraphState) -> GraphState:
        """
        Node 1: Parse email and extract attachments.
        
        Input: email_path
        Output: email_headers, email_body, attachments
        """
        logger.info(f"[PREPROCESS] Processing email: {state.email_path}")
        state.start_time = datetime.utcnow()
        
        try:
            parser = EmailParser()
            email_data = await parser.parse(state.email_path)
            
            state.email_headers = email_data['headers']
            state.email_body = email_data['body']
            state.attachments = email_data['attachments']
            
            # Create ParsedContent for email body
            state.parsed_contents.append(
                ParsedContent(
                    source_type=SourceType.EMAIL_BODY,
                    content=state.email_body,
                    metadata={'headers': state.email_headers}
                )
            )
            
            logger.info(
                f"[PREPROCESS] Extracted {len(state.attachments)} attachments"
            )
            
        except Exception as e:
            error = ErrorRecord(
                error_type="PreprocessError",
                error_message=str(e),
                component="preprocess",
                recoverable=False
            )
            state.errors.append(error)
            logger.error(f"[PREPROCESS] Error: {e}")
        
        return state
    
    async def _parse_documents_node(self, state: GraphState) -> GraphState:
        """
        Node 2: Parse attachments using external API.
        
        Input: attachments
        Output: parsed_contents (extended)
        """
        logger.info(f"[PARSE] Parsing {len(state.attachments)} attachments")
        
        if not state.attachments:
            logger.info("[PARSE] No attachments to parse")
            return state
        
        try:
            async with DocumentParser(self.config.parser) as parser:
                parsed = await parser.parse_batch(state.attachments)
                state.parsed_contents.extend(parsed)
            
            logger.info(f"[PARSE] Parsed {len(parsed)} content blocks")
            
        except Exception as e:
            error = ErrorRecord(
                error_type="ParseError",
                error_message=str(e),
                component="parse_documents",
                recoverable=True
            )
            state.errors.append(error)
            logger.error(f"[PARSE] Error: {e}")
        
        return state
    
    async def _chunk_content_node(self, state: GraphState) -> GraphState:
        """
        Node 3: Chunk all parsed content.
        
        Input: parsed_contents
        Output: chunks
        """
        logger.info(f"[CHUNK] Chunking {len(state.parsed_contents)} content blocks")
        
        try:
            chunker = TextChunker(self.config.chunking)
            state.chunks = chunker.chunk_all(state.parsed_contents)
            
            logger.info(f"[CHUNK] Created {len(state.chunks)} chunks")
            
        except Exception as e:
            error = ErrorRecord(
                error_type="ChunkError",
                error_message=str(e),
                component="chunk_content",
                recoverable=False
            )
            state.errors.append(error)
            logger.error(f"[CHUNK] Error: {e}")
        
        return state
    
    async def _parallel_extract_node(self, state: GraphState) -> GraphState:
        """
        Node 4: Run parallel category agents for extraction.
        
        Input: chunks, field_metadata
        Output: agent_states
        """
        is_retry = state.retry_count > 0
        
        logger.info(
            f"[EXTRACT] Running parallel extraction "
            f"(retry: {is_retry}, attempt: {state.retry_count + 1})"
        )
        
        try:
            async with LLMAdapter(self.config.llm) as llm:
                orchestrator = AgentOrchestrator(llm)
                
                # Create agents if first run
                if not state.agent_states:
                    agents = orchestrator.create_agents(state.field_metadata)
                    state.agent_states = {
                        cat: agent.state for cat, agent in agents.items()
                    }
                
                # Run extraction
                agent_states = await orchestrator.run_all_agents(
                    state.chunks,
                    is_retry
                )
                
                # Update state
                state.agent_states.update(agent_states)
            
            logger.info("[EXTRACT] Parallel extraction completed")
            
        except Exception as e:
            error = ErrorRecord(
                error_type="ExtractError",
                error_message=str(e),
                component="parallel_extract",
                recoverable=True
            )
            state.errors.append(error)
            logger.error(f"[EXTRACT] Error: {e}")
        
        return state
    
    async def _aggregate_node(self, state: GraphState) -> GraphState:
        """
        Node 5: Aggregate results from all agents.
        
        Input: agent_states
        Output: extracted_fields, missing_fields
        """
        logger.info("[AGGREGATE] Aggregating agent results")
        
        try:
            # Merge extracted fields from all agents
            for agent_state in state.agent_states.values():
                state.extracted_fields.update(agent_state.extracted_fields)
            
            # Identify missing fields
            all_field_names = {f.name for f in state.field_metadata}
            extracted_names = set(state.extracted_fields.keys())
            state.missing_fields = list(all_field_names - extracted_names)
            
            fill_rate = len(extracted_names) / len(all_field_names) if all_field_names else 0
            
            logger.info(
                f"[AGGREGATE] Extracted {len(extracted_names)}/{len(all_field_names)} "
                f"fields ({fill_rate:.1%} fill rate)"
            )
            
            if state.missing_fields:
                logger.info(f"[AGGREGATE] Missing fields: {state.missing_fields}")
        
        except Exception as e:
            error = ErrorRecord(
                error_type="AggregateError",
                error_message=str(e),
                component="aggregate",
                recoverable=True
            )
            state.errors.append(error)
            logger.error(f"[AGGREGATE] Error: {e}")
        
        return state
    
    def _should_retry(self, state: GraphState) -> str:
        """
        Conditional edge: Determine if retry is needed.
        
        Returns:
            "retry" if conditions are met, "finalize" otherwise
        """
        # Check retry conditions
        has_missing = len(state.missing_fields) > 0
        under_limit = state.retry_count < state.max_retries
        has_chunks = len(state.chunks) > 0
        
        if has_missing and under_limit and has_chunks:
            logger.info(
                f"[DECISION] Retry needed: {len(state.missing_fields)} missing fields, "
                f"attempt {state.retry_count + 1}/{state.max_retries}"
            )
            return "retry"
        else:
            logger.info("[DECISION] Proceeding to finalization")
            return "finalize"
    
    async def _retry_extraction_node(self, state: GraphState) -> GraphState:
        """
        Node 6: Retry extraction for missing fields.
        
        Input: missing_fields, chunks
        Output: Updated agent_states
        """
        logger.info(f"[RETRY] Retrying {len(state.missing_fields)} missing fields")
        
        state.retry_count += 1
        state.retry_fields = state.missing_fields.copy()
        
        # The actual retry happens in parallel_extract with is_retry=True
        # This node just updates retry tracking
        
        return state
    
    async def _finalize_node(self, state: GraphState) -> GraphState:
        """
        Node 7: Compute metrics and finalize output.
        
        Input: extracted_fields, agent_states, errors
        Output: metrics
        """
        logger.info("[FINALIZE] Computing metrics and finalizing output")
        
        try:
            state.end_time = datetime.utcnow()
            
            # Calculate metrics
            total_fields = len(state.field_metadata)
            extracted_count = len(state.extracted_fields)
            fill_rate = extracted_count / total_fields if total_fields > 0 else 0.0
            
            # Average confidence
            confidences = [f.confidence for f in state.extracted_fields.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Processing time
            processing_time = 0.0
            if state.start_time and state.end_time:
                delta = state.end_time - state.start_time
                processing_time = delta.total_seconds()
            
            # Agent times
            agent_times = {}
            for cat, agent_state in state.agent_states.items():
                if agent_state.start_time and agent_state.end_time:
                    delta = agent_state.end_time - agent_state.start_time
                    agent_times[cat] = delta.total_seconds()
                else:
                    agent_times[cat] = 0.0
            
            # Retry lift
            initial_extracted = extracted_count
            if state.retry_count > 0:
                # Estimate initial extraction
                retry_fields = set(state.retry_fields)
                initial_extracted = sum(
                    1 for name in state.extracted_fields.keys()
                    if name not in retry_fields
                )
            
            retry_lift = 0.0
            if state.retry_count > 0 and initial_extracted < extracted_count:
                retry_lift = (extracted_count - initial_extracted) / total_fields
            
            # Total chunks scanned
            total_chunks_scanned = sum(
                agent.chunks_scanned
                for agent in state.agent_states.values()
            )
            
            state.metrics = ExtractionMetrics(
                total_fields=total_fields,
                extracted_fields=extracted_count,
                fill_rate=fill_rate,
                avg_confidence=avg_confidence,
                retry_lift=retry_lift,
                total_chunks=len(state.chunks),
                chunks_scanned=total_chunks_scanned,
                processing_time_seconds=processing_time,
                agent_times=agent_times,
                errors_count=len(state.errors)
            )
            
            logger.info(
                f"[FINALIZE] Metrics: {extracted_count}/{total_fields} fields "
                f"({fill_rate:.1%}), avg confidence: {avg_confidence:.2f}, "
                f"time: {processing_time:.1f}s"
            )
            
        except Exception as e:
            error = ErrorRecord(
                error_type="FinalizeError",
                error_message=str(e),
                component="finalize",
                recoverable=False
            )
            state.errors.append(error)
            logger.error(f"[FINALIZE] Error: {e}")
        
        return state
    
    async def run(self, input_data: Dict[str, Any]) -> GraphState:
        """
        Execute the workflow.
        
        Args:
            input_data: Must contain:
                - email_path: Path to email file
                - document_id: Unique identifier
                - field_metadata: List of FieldMetadata objects
        
        Returns:
            Final GraphState with extracted fields and metrics
        """
        # Initialize state
        initial_state = GraphState(
            document_id=input_data['document_id'],
            email_path=input_data['email_path'],
            field_metadata=input_data.get('field_metadata', []),
            max_retries=self.config.retry.max_retries
        )
        
        # Execute graph
        logger.info(f"Starting extraction workflow for {input_data['document_id']}")
        
        final_state = await self.graph.ainvoke(initial_state)
        
        logger.info(f"Workflow completed for {input_data['document_id']}")
        
        return final_state


def build_extraction_graph(config: ExtractionConfig) -> ExtractionWorkflow:
    """
    Factory function to build the extraction workflow.
    
    Args:
        config: Extraction configuration
    
    Returns:
        Compiled ExtractionWorkflow
    """
    return ExtractionWorkflow(config)
