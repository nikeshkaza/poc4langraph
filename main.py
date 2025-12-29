"""
Main entry point for the email extraction system.
"""

import asyncio
import sys
from pathlib import Path
import yaml
import logging

from src.models.config import ExtractionConfig
from src.models.schemas import FieldMetadata
from src.graph.workflow import build_extraction_graph
from src.utils.helpers import (
    setup_logging,
    format_extraction_result,
    export_result_to_json,
    print_extraction_summary
)

logger = logging.getLogger(__name__)


def load_field_metadata(path: str) -> list:
    """Load field definitions from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    fields = []
    for field_data in data.get('fields', []):
        fields.append(FieldMetadata(**field_data))
    
    return fields


async def extract_from_email(
    email_path: str,
    document_id: str,
    config_path: str = "config/extraction_config.yaml",
    output_path: str = None
) -> dict:
    """
    Extract fields from an email file.
    
    Args:
        email_path: Path to .eml or .msg file
        document_id: Unique identifier for this document
        config_path: Path to configuration YAML
        output_path: Optional path to save results JSON
    
    Returns:
        Extraction results as dict
    """
    # Load configuration
    config = ExtractionConfig.from_yaml(config_path)
    setup_logging(config)
    
    logger.info(f"Starting extraction for document: {document_id}")
    logger.info(f"Email path: {email_path}")
    
    # Load field metadata
    field_metadata = load_field_metadata(config.field_metadata_path)
    logger.info(f"Loaded {len(field_metadata)} field definitions")
    
    # Build and run workflow
    workflow = build_extraction_graph(config)
    
    final_state = await workflow.run({
        'email_path': email_path,
        'document_id': document_id,
        'field_metadata': field_metadata
    })
    
    # Format result
    result = format_extraction_result(final_state)
    
    # Export to file if requested
    if output_path:
        export_result_to_json(result, output_path)
    
    # Print summary
    print_extraction_summary(result)
    
    return result.model_dump() if hasattr(result, 'model_dump') else result.dict()


async def main():
    """Main CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python main.py <email_path> <document_id> [output_path]")
        print("\nExample:")
        print("  python main.py examples/sample_email.eml DOC-001 results/output.json")
        sys.exit(1)
    
    email_path = sys.argv[1]
    document_id = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Validate email file exists
    if not Path(email_path).exists():
        print(f"Error: Email file not found: {email_path}")
        sys.exit(1)
    
    try:
        result = await extract_from_email(
            email_path=email_path,
            document_id=document_id,
            output_path=output_path
        )
        
        print(f"\n✅ Extraction completed successfully!")
        
        if output_path:
            print(f"📄 Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        print(f"\n❌ Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
