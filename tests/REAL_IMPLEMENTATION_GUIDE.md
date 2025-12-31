# Real Implementation Setup Guide

## Overview

This system now uses **real** document parsing and LLM extraction:
- **PyMuPDF** for PDF extraction
- **Azure Document Intelligence** for DOCX, DOC, and images (OCR)
- **OpenAI GPT-4** for field extraction

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `PyMuPDF` - PDF text extraction
- `azure-ai-formrecognizer` - Azure Document Intelligence SDK
- `openai` - OpenAI GPT API client
- All other dependencies

### 2. Azure Document Intelligence Setup

#### Create Azure Resource

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new "Document Intelligence" resource (formerly Form Recognizer)
3. Note your:
   - **Endpoint**: `https://your-resource.cognitiveservices.azure.com/`
   - **API Key**: Found in "Keys and Endpoint" section

#### Set Environment Variables

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key-here
```

### 3. OpenAI API Setup

#### Get API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys
3. Create a new API key

#### Set Environment Variable

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Or add to `.env`:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

## Configuration

### Update config/extraction_config.yaml

```yaml
# LLM Configuration
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"  # or gpt-4, gpt-3.5-turbo
  temperature: 0.0
  max_tokens: 2000
  timeout: 30

# No parser endpoint needed - using Azure SDK directly
parser:
  endpoint: "https://not-used.com"  # Not used with Azure SDK
  timeout: 120  # Increase for large documents
  supported_types:
    - pdf
    - docx
    - doc
    - png
    - jpg
    - jpeg
```

## How It Works

### Document Parsing Flow

```
Attachment → Check Type
    │
    ├─ PDF → PyMuPDF
    │         ├─ Extract text per page
    │         ├─ Build line map
    │         └─ Return ParsedContent[]
    │
    ├─ DOCX/DOC → Azure Document Intelligence
    │              ├─ Upload to Azure
    │              ├─ Analyze with prebuilt-read model
    │              ├─ Extract text per page
    │              └─ Return ParsedContent[]
    │
    └─ Image → Azure Document Intelligence (OCR)
                ├─ Upload to Azure
                ├─ OCR with prebuilt-read model
                ├─ Extract text
                └─ Return ParsedContent[]
```

### Field Extraction Flow

```
ParsedContent[] → Chunking → Chunks[]
                                │
                                ↓
                    5 Parallel Category Agents
                                │
                    Each agent for their fields:
                                │
                    ↓
        Build prompt with field metadata + chunks
                    ↓
        Call OpenAI GPT-4 (REAL API)
                    ↓
        Parse JSON response
                    ↓
        Extract: {
            "found": true/false,
            "value": "extracted value",
            "confidence": 0.92,
            "chunk_id": "uuid",
            "reasoning": "explanation"
        }
                    ↓
        Create ExtractedField with Citation
                    ↓
        Return to aggregator
```

## Usage Examples

### Basic Usage

```bash
# Make sure environment variables are set
export OPENAI_API_KEY="sk-..."
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://..."
export AZURE_DOCUMENT_INTELLIGENCE_KEY="..."

# Run extraction
python main.py examples/sample_email.eml DOC-001 results/output.json
```

### Python API

```python
import asyncio
from main import extract_from_email

async def main():
    result = await extract_from_email(
        email_path="path/to/email.eml",
        document_id="DOC-12345",
        output_path="results/output.json"
    )
    
    print(f"Extracted {len(result['fields'])} fields")
    print(f"Fill rate: {result['metrics']['fill_rate']:.1%}")

asyncio.run(main())
```

### Processing Multiple Emails

```python
import asyncio
from main import extract_from_email

async def process_batch(emails):
    tasks = []
    for i, email_path in enumerate(emails):
        task = extract_from_email(
            email_path=email_path,
            document_id=f"DOC-{i+1:04d}",
            output_path=f"results/output_{i+1}.json"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# Process
emails = ["email1.eml", "email2.eml", "email3.eml"]
results = asyncio.run(process_batch(emails))
```

## Output Structure

The system returns detailed extraction results:

```json
{
  "document_id": "DOC-001",
  "fields": {
    "company_name": {
      "value": "Acme Corporation",
      "confidence": 0.95,
      "citation": {
        "source": "attachment",
        "attachment_name": "contract.pdf",
        "chunk_id": "uuid-123",
        "page": 1,
        "line_numbers": [5, 6],
        "extraction_pass": "initial"
      }
    }
  },
  "metrics": {
    "total_fields": 50,
    "extracted_fields": 42,
    "fill_rate": 0.84,
    "avg_confidence": 0.89,
    "processing_time_seconds": 45.2,
    "parser_metadata": {
      "contract.pdf": {
        "page_count": 12,
        "total_characters": 45230,
        "parser_used": "pymupdf"
      },
      "invoice.docx": {
        "page_count": 3,
        "total_characters": 8940,
        "parser_used": "azure_document_intelligence"
      }
    }
  }
}
```

## Real Extraction Details

### PDF Parsing with PyMuPDF

**Features**:
- Fast and accurate text extraction
- Page-by-page processing
- Line number mapping
- Metadata: page dimensions, page count

**What it extracts**:
```python
{
    'source_type': 'attachment',
    'attachment_name': 'document.pdf',
    'page': 1,
    'content': 'Full text from page 1...',
    'line_map': [(1, 'Line 1 text'), (2, 'Line 2 text')],
    'metadata': {
        'page_count': 10,
        'page_width': 612.0,
        'page_height': 792.0,
        'parser': 'pymupdf'
    }
}
```

### Azure Document Intelligence

**Features**:
- OCR for images and scanned documents
- Layout analysis
- Multi-language support
- Table extraction (can be enabled)

**What it extracts**:
```python
{
    'source_type': 'attachment',
    'attachment_name': 'invoice.docx',
    'page': 1,
    'content': 'Full text from page 1...',
    'line_map': [(1, 'Line 1'), (2, 'Line 2')],
    'metadata': {
        'page_count': 3,
        'page_width': 8.5,
        'page_height': 11.0,
        'parser': 'azure_document_intelligence',
        'model_id': 'prebuilt-read'
    }
}
```

### OpenAI GPT-4 Extraction

**How it works**:
1. System prompt defines extraction rules
2. User prompt includes:
   - Field name and description
   - Field examples
   - All relevant chunks
3. GPT-4 returns structured JSON:
   ```json
   {
       "found": true,
       "value": "Extracted value",
       "confidence": 0.92,
       "chunk_id": "chunk-uuid",
       "reasoning": "Found in section 3, line 45"
   }
   ```

**Model Options**:
- `gpt-4-turbo-preview` (Recommended) - Best accuracy
- `gpt-4` - High accuracy
- `gpt-3.5-turbo` - Faster, lower cost, slightly less accurate

## Troubleshooting

### PyMuPDF Issues

**Error**: `ModuleNotFoundError: No module named 'fitz'`
```bash
pip install PyMuPDF
```

**Error**: `Failed to open PDF`
- Check file is valid PDF
- Try: `fitz.open(pdf_path)` in Python shell

### Azure Issues

**Error**: `Azure client not initialized`
```bash
# Check environment variables
echo $AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
echo $AZURE_DOCUMENT_INTELLIGENCE_KEY

# Should output your values
```

**Error**: `Authentication failed`
- Verify API key is correct
- Check endpoint URL format: `https://your-resource.cognitiveservices.azure.com/`

**Error**: `Quota exceeded`
- Check Azure pricing tier
- Consider upgrading or rate limiting

### OpenAI Issues

**Error**: `OpenAI API key not found`
```bash
export OPENAI_API_KEY="sk-..."
```

**Error**: `Rate limit exceeded`
- Implement exponential backoff (already included)
- Reduce concurrent requests
- Upgrade API tier

**Error**: `Token limit exceeded`
- Reduce chunk size in config
- Use fewer chunks per field
- Split large documents

## Cost Estimation

### Azure Document Intelligence
- **Free tier**: 500 pages/month
- **Paid**: $1.50 per 1000 pages (prebuilt-read)

### OpenAI GPT-4
- **Input**: ~$10 per 1M tokens
- **Output**: ~$30 per 1M tokens
- Typical email: 50 fields × 5 chunks × 1000 tokens = ~250K tokens
- Cost per email: ~$2.50-$5.00

### Cost Reduction Tips
1. Use `gpt-3.5-turbo` for non-critical fields ($0.50 per 1M tokens)
2. Implement chunk caching
3. Reduce max_tokens in config
4. Use early stopping (already implemented)

## Performance Benchmarks

**Typical Performance**:
- PDF (10 pages): 2-5 seconds (PyMuPDF)
- DOCX (5 pages): 5-10 seconds (Azure)
- Image OCR: 3-8 seconds per image (Azure)
- GPT extraction: 1-3 seconds per field
- **Total for 1 email + 3 attachments**: 30-60 seconds

**Parallel Processing**:
- 5 agents run concurrently
- 3-5x speedup vs sequential

## Next Steps

1. ✅ Set up environment variables
2. ✅ Install dependencies
3. ✅ Test with sample email
4. ✅ Monitor API costs
5. ✅ Optimize chunk sizes
6. ✅ Scale with worker pools

## Support

For issues:
- PyMuPDF: https://pymupdf.readthedocs.io/
- Azure Docs: https://learn.microsoft.com/azure/ai-services/document-intelligence/
- OpenAI Docs: https://platform.openai.com/docs/
