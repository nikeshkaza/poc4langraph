# LangGraph Multi-Agent Email & Attachment Field Extraction System

## Architecture Overview

This is a production-grade, multi-agent extraction pipeline built with LangGraph for structured field extraction from emails and attachments.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                │
│                    (.eml / .msg + Attachments)                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      PRE-PROCESSING NODE                             │
│  • Parse email headers/body                                         │
│  • Extract attachments                                              │
│  • Route by file type                                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PARSING NODE                             │
│  • Call external parser API per attachment                          │
│  • Handle timeouts/failures gracefully                              │
│  • Normalize to unified schema                                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     CHUNKING & INDEXING NODE                         │
│  • Semantic chunking by source                                      │
│  • Preserve provenance metadata                                     │
│  • Build searchable index                                           │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  PARALLEL CATEGORY AGENTS (5)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent 4  │  ...       │
│  │Category A│  │Category B│  │Category C│  │Category D│            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│  • Field-by-field extraction                                        │
│  • Confidence scoring                                               │
│  • Citation tracking                                                │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    AGGREGATION NODE                                  │
│  • Merge results from all agents                                    │
│  • Identify missing fields                                          │
│  • Compute initial metrics                                          │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                    ┌───────────────────┐
                    │  All fields filled?│
                    └───────────────────┘
                       ↙              ↘
                    YES               NO
                     ↓                 ↓
            ┌─────────────┐   ┌─────────────────┐
            │   OUTPUT    │   │  RETRY NODE     │
            │  ASSEMBLY   │   │  • Focused      │
            └─────────────┘   │    re-extraction│
                              │  • Relaxed      │
                              │    prompts      │
                              └─────────────────┘
                                      ↓
                              ┌─────────────────┐
                              │  RE-AGGREGATION │
                              │  • Update state │
                              │  • Track lift   │
                              └─────────────────┘
                                      ↓
                              ┌─────────────────┐
                              │ Max retries OR  │
                              │ all filled?     │
                              └─────────────────┘
                                      ↓
                              ┌─────────────────┐
                              │ OUTPUT ASSEMBLY │
                              └─────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT                                 │
│  {                                                                   │
│    "document_id": "...",                                            │
│    "fields": {...},                                                 │
│    "metrics": {...},                                                │
│    "errors": [...]                                                  │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Metadata-Driven**: No hardcoded fields, fully configurable
- **Parallel Processing**: 5 concurrent category agents
- **Provenance Tracking**: Full citation chain for every extracted value
- **Retry Logic**: Smart re-extraction for missing fields
- **Error Isolation**: Per-agent exception handling
- **Production-Ready**: Comprehensive logging, metrics, and monitoring
- **Async-First**: Non-blocking I/O operations
- **Type-Safe**: Full type hints and validation

## Project Structure

```
email-extraction-system/
├── src/
│   ├── models/           # Pydantic models & schemas
│   ├── parsers/          # Email & document parsing adapters
│   ├── agents/           # Category extraction agents
│   ├── graph/            # LangGraph workflow definition
│   └── utils/            # Helpers, logging, metrics
├── config/               # Configuration files
├── tests/                # Unit & integration tests
├── examples/             # Sample inputs & outputs
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.graph.workflow import build_extraction_graph
from src.models.config import ExtractionConfig

# Load configuration
config = ExtractionConfig.from_yaml("config/extraction_config.yaml")

# Build the graph
graph = build_extraction_graph(config)

# Execute extraction
result = await graph.ainvoke({
    "email_path": "path/to/email.eml",
    "document_id": "DOC-12345"
})

print(result["fields"])
print(result["metrics"])
```

## Configuration

See `config/extraction_config.yaml` for full configuration options including:
- Field metadata definitions
- Chunking parameters
- Retry settings
- API endpoints
- Logging levels

## Scalability Considerations

1. **Horizontal Scaling**: Each category agent can run on separate workers
2. **Caching**: Implement Redis/Memcached for parsed documents
3. **Batching**: Process multiple emails in parallel
4. **Rate Limiting**: Built-in backoff for external API calls
5. **Monitoring**: Export metrics to Prometheus/Datadog

## Future Extensions

- Vector embeddings for semantic search
- Active learning for field discovery
- Multi-language support
- Real-time streaming extraction
- Human-in-the-loop validation UI

## License

MIT
