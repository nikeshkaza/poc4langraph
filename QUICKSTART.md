# Quick Start Guide

## Installation

### 1. Clone and Install Dependencies

```bash
cd email-extraction-system
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export PARSER_API_ENDPOINT="https://your-parser-api.com"
export PARSER_API_KEY="your-parser-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
PARSER_API_ENDPOINT=https://your-parser-api.com
PARSER_API_KEY=your-parser-api-key
```

### 3. Configure Field Definitions

Edit `config/field_definitions.yaml` to define your extraction fields:

```yaml
fields:
  - name: "company_name"
    description: "Legal name of the company"
    category: "company_info"
    field_type: "string"
    required: true
    priority: 5
```

### 4. Run Extraction

```bash
python main.py examples/sample_email.eml DOC-001 results/output.json
```

## Basic Usage

### Python API

```python
from src.graph.workflow import build_extraction_graph
from src.models.config import ExtractionConfig
import asyncio

async def main():
    # Load configuration
    config = ExtractionConfig.from_yaml("config/extraction_config.yaml")
    
    # Load field metadata
    from main import load_field_metadata
    fields = load_field_metadata(config.field_metadata_path)
    
    # Build workflow
    workflow = build_extraction_graph(config)
    
    # Run extraction
    result = await workflow.run({
        'email_path': 'path/to/email.eml',
        'document_id': 'DOC-12345',
        'field_metadata': fields
    })
    
    # Access results
    print(f"Extracted {len(result.extracted_fields)} fields")
    print(f"Fill rate: {result.metrics.fill_rate:.1%}")

asyncio.run(main())
```

### Batch Processing

```python
import asyncio
from main import extract_from_email

async def process_batch(email_paths):
    tasks = [
        extract_from_email(path, f"DOC-{i}")
        for i, path in enumerate(email_paths)
    ]
    results = await asyncio.gather(*tasks)
    return results

# Process multiple emails
emails = ['email1.eml', 'email2.eml', 'email3.eml']
results = asyncio.run(process_batch(emails))
```

## Configuration

### LLM Provider

**OpenAI**:
```yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  api_key: null  # Set via environment
```

**Anthropic Claude**:
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"
  api_key: null
```

**Azure OpenAI**:
```yaml
llm:
  provider: "azure"
  endpoint: "https://your-resource.openai.azure.com"
  model: "gpt-4"
  api_key: null
```

### Chunking Parameters

```yaml
chunking:
  chunk_size: 1000      # Characters per chunk
  chunk_overlap: 200    # Overlap between chunks
  min_chunk_size: 100   # Minimum chunk size
  separator: "\n\n"     # Preferred split point
```

### Retry Configuration

```yaml
retry:
  max_retries: 2                      # Maximum retry attempts
  retry_missing_only: true            # Only retry missing fields
  relaxed_confidence_threshold: 0.5   # Lower threshold on retry
  use_synonyms: true                  # Use field synonyms
  expand_context: true                # Include surrounding chunks
```

## Testing

### Run Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_extraction.py::TestChunking::test_chunk_long_text -v
```

## Common Use Cases

### 1. Contract Processing

Extract: contract number, dates, parties, values, terms

```yaml
fields:
  - name: "contract_number"
    category: "contract"
    priority: 5
  - name: "contract_value"
    category: "contract"
    priority: 5
```

### 2. Invoice Extraction

Extract: invoice number, date, amounts, line items

```yaml
fields:
  - name: "invoice_number"
    category: "financial"
  - name: "total_amount"
    category: "financial"
```

### 3. Resume/CV Parsing

Extract: name, contact, experience, education

```yaml
fields:
  - name: "candidate_name"
    category: "personal"
  - name: "years_experience"
    category: "professional"
```

### 4. Legal Document Review

Extract: parties, dates, obligations, jurisdictions

```yaml
fields:
  - name: "governing_law"
    category: "legal"
  - name: "termination_clause"
    category: "legal"
```

## Troubleshooting

### Issue: Low Fill Rate

**Solution**:
1. Increase retry attempts
2. Add more field examples
3. Use broader synonyms
4. Increase chunk size for more context

### Issue: Low Confidence Scores

**Solution**:
1. Improve field descriptions
2. Add validation examples
3. Use higher quality LLM
4. Reduce chunk size for precision

### Issue: Slow Processing

**Solution**:
1. Enable caching
2. Reduce max_retries
3. Use smaller LLM for retry
4. Process in parallel batches

### Issue: Memory Usage

**Solution**:
1. Reduce chunk_size
2. Process fewer documents concurrently
3. Enable streaming mode
4. Clear cache periodically

## Best Practices

### Field Definitions

✅ **Good**:
```yaml
- name: "contract_date"
  description: "The date when the contract was signed, typically in MM/DD/YYYY format"
  examples: ["01/15/2024", "2024-01-15"]
  synonyms: ["signature date", "execution date"]
```

❌ **Bad**:
```yaml
- name: "date"
  description: "A date"
  # Too vague, no examples
```

### Prompting

✅ **Good**: Clear, specific field descriptions with examples

❌ **Bad**: Vague descriptions without context

### Error Handling

✅ **Good**: Log errors but continue processing

❌ **Bad**: Fail entire extraction on single error

### Validation

✅ **Good**: Define validation rules for critical fields

```yaml
validation_rules:
  - rule_type: "regex"
    value: "^\\d{2}-\\d{7}$"
```

❌ **Bad**: No validation, accept any value

## Performance Tips

1. **Use GPU for OCR**: Speeds up image processing 10x
2. **Cache Parsed Documents**: Avoid re-parsing same attachments
3. **Batch LLM Calls**: Combine multiple fields in one call
4. **Early Stopping**: Stop scanning chunks after confident match
5. **Parallel Agents**: Already implemented, scales linearly

## Support

For issues or questions:
- Check ARCHITECTURE.md for system details
- Check DEPLOYMENT.md for scaling guidance
- Review logs in `logs/extraction.log`
- Enable DEBUG logging for detailed traces

## Next Steps

1. Customize field definitions for your use case
2. Integrate with your document parser API
3. Set up monitoring and metrics
4. Deploy to production (see DEPLOYMENT.md)
5. Optimize based on your workload
