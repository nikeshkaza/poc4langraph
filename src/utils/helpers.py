"""
Utility functions for logging, metrics, and helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from src.models.schemas import ExtractionResult, GraphState
from src.models.config import ExtractionConfig


def setup_logging(config: ExtractionConfig) -> None:
    """
    Configure logging for the application.
    
    Args:
        config: Extraction configuration with log settings
    """
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.log_to_file and config.log_path:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.log_path))
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set library log levels
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def format_extraction_result(state: GraphState) -> ExtractionResult:
    """
    Convert GraphState to final ExtractionResult format.
    
    Args:
        state: Final graph state
    
    Returns:
        Formatted extraction result
    """
    return ExtractionResult(
        document_id=state.document_id,
        email_subject=state.email_headers.get('subject'),
        fields=state.extracted_fields,
        metrics=state.metrics,
        errors=state.errors,
        timestamp=state.end_time or datetime.utcnow()
    )


def export_result_to_json(result: ExtractionResult, output_path: str) -> None:
    """
    Export extraction result to JSON file.
    
    Args:
        result: Extraction result to export
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result.dict(), f, indent=2, default=str)
    
    logging.info(f"Result exported to {output_path}")


def print_extraction_summary(result: ExtractionResult) -> None:
    """
    Print a human-readable summary of extraction results.
    
    Args:
        result: Extraction result to summarize
    """
    print("\n" + "="*80)
    print(f"EXTRACTION SUMMARY - Document: {result.document_id}")
    print("="*80)
    
    if result.email_subject:
        print(f"Email Subject: {result.email_subject}")
    
    print(f"\n📊 METRICS:")
    print(f"  • Fields Extracted: {result.metrics.extracted_fields}/{result.metrics.total_fields}")
    print(f"  • Fill Rate: {result.metrics.fill_rate:.1%}")
    print(f"  • Avg Confidence: {result.metrics.avg_confidence:.2f}")
    print(f"  • Processing Time: {result.metrics.processing_time_seconds:.1f}s")
    
    if result.metrics.retry_lift > 0:
        print(f"  • Retry Lift: +{result.metrics.retry_lift:.1%}")
    
    print(f"  • Total Chunks: {result.metrics.total_chunks}")
    print(f"  • Chunks Scanned: {result.metrics.chunks_scanned}")
    
    if result.metrics.errors_count > 0:
        print(f"  ⚠️  Errors: {result.metrics.errors_count}")
    
    print(f"\n📝 EXTRACTED FIELDS:")
    for name, field in sorted(result.fields.items()):
        confidence_icon = "✓" if field.confidence >= 0.8 else "~"
        print(f"  {confidence_icon} {name}: {field.value}")
        print(f"     └─ Source: {field.citation.source.value}", end="")
        if field.citation.attachment_name:
            print(f" ({field.citation.attachment_name})", end="")
        if field.citation.page:
            print(f", Page {field.citation.page}", end="")
        print(f" [Confidence: {field.confidence:.2f}]")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for error in result.errors:
            print(f"  • [{error.component}] {error.error_type}: {error.error_message}")
    
    print("\n" + "="*80 + "\n")


class MetricsTracker:
    """Track extraction metrics across multiple documents."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: ExtractionResult) -> None:
        """Add a result to tracking."""
        self.results.append(result)
    
    def get_aggregate_metrics(self) -> dict:
        """Compute aggregate metrics across all results."""
        if not self.results:
            return {}
        
        total_fields = sum(r.metrics.total_fields for r in self.results)
        extracted_fields = sum(r.metrics.extracted_fields for r in self.results)
        
        return {
            'total_documents': len(self.results),
            'total_fields': total_fields,
            'total_extracted': extracted_fields,
            'avg_fill_rate': extracted_fields / total_fields if total_fields > 0 else 0,
            'avg_confidence': sum(r.metrics.avg_confidence for r in self.results) / len(self.results),
            'avg_processing_time': sum(r.metrics.processing_time_seconds for r in self.results) / len(self.results),
            'total_errors': sum(r.metrics.errors_count for r in self.results)
        }
    
    def export_aggregate(self, output_path: str) -> None:
        """Export aggregate metrics to JSON."""
        metrics = self.get_aggregate_metrics()
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)


def validate_field_value(field, validation_rules) -> tuple[bool, list]:
    """
    Validate extracted field against rules.
    
    Args:
        field: Extracted field value
        validation_rules: List of ValidationRule objects
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    for rule in validation_rules:
        if rule.rule_type == 'regex':
            import re
            if not re.match(rule.value, str(field)):
                errors.append(rule.error_message or f"Value doesn't match pattern {rule.value}")
        
        elif rule.rule_type == 'range':
            min_val, max_val = rule.value
            try:
                val = float(field)
                if not (min_val <= val <= max_val):
                    errors.append(rule.error_message or f"Value must be between {min_val} and {max_val}")
            except ValueError:
                errors.append("Value must be numeric")
        
        elif rule.rule_type == 'enum':
            if field not in rule.value:
                errors.append(rule.error_message or f"Value must be one of {rule.value}")
    
    return len(errors) == 0, errors
