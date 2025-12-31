# Architecture Deep Dive

## System Components

### 1. Pre-Processing Layer

**Responsibility**: Parse email files and extract attachments

**Components**:
- `EmailParser`: Handles .eml and .msg formats
- Extracts headers, body, and attachment metadata
- Routes attachments by type (PDF, DOCX, Image, Other)

**Error Handling**:
- Graceful fallback for corrupt emails
- HTML-to-text conversion for rich emails
- Unsupported format detection

### 2. Document Parsing Layer

**Responsibility**: Convert attachments to normalized text

**Components**:
- `DocumentParser`: Adapter for external parsing API
- Handles: PDF, DOCX, DOC, Images (OCR)
- Implements retry logic with exponential backoff

**Features**:
- Async/concurrent parsing
- Per-attachment error isolation
- Circuit breaker for API failures
- Response normalization to `ParsedContent` schema

**Integration Points**:
- Azure Form Recognizer
- PyMuPDF
- Tesseract OCR
- Custom parsing APIs

### 3. Chunking & Indexing Layer

**Responsibility**: Break text into searchable chunks

**Components**:
- `TextChunker`: Semantic text segmentation
- Configurable chunk size and overlap
- Provenance preservation (source, page, line numbers)

**Algorithm**:
```
For each ParsedContent:
  1. If text < min_size: create single chunk
  2. Else:
     - Iterate through text with sliding window
     - Break at semantic boundaries (paragraphs)
     - Maintain overlap for context
     - Preserve metadata (source, page, lines)
  3. Assign unique chunk_id
  4. Build searchable index
```

**Output**: List of `Chunk` objects with full citation chain

### 4. Parallel Category Agents

**Responsibility**: Extract fields by category

**Architecture**:
```
AgentOrchestrator
  ├── Agent 1: Company Info (10 fields)
  ├── Agent 2: Financial (10 fields)
  ├── Agent 3: Contract (10 fields)
  ├── Agent 4: Compliance (10 fields)
  └── Agent 5: Personnel (10 fields)
```

**Agent Workflow**:
```
For each assigned field:
  1. Build extraction prompt with field metadata
  2. Iterate through chunks (early stopping on match)
  3. Call LLM with structured output format
  4. Parse response → ExtractedField
  5. Validate against rules
  6. Update agent state
  7. Track confidence and provenance
```

**Parallelization**:
- All agents run concurrently via `asyncio.gather`
- Independent state management
- Isolated error handling (one failure doesn't stop others)

### 5. Retry & Refinement Layer

**Responsibility**: Re-extract missing fields

**Strategy**:
```
If missing fields exist AND retries < max:
  1. Identify missing fields
  2. Build focused prompts with:
     - Synonyms
     - Relaxed confidence threshold
     - Expanded context (surrounding chunks)
  3. Re-run only affected agents
  4. Track retry metrics (lift %)
```

**Optimizations**:
- Only retry high-priority fields
- Use chunk relevance scoring
- Apply semantic similarity search
- Lower temperature for creativity

### 6. Aggregation Layer

**Responsibility**: Merge results from all agents

**Process**:
```
1. Collect extracted fields from all agent states
2. Merge into single dictionary
3. Identify missing fields
4. Compute initial metrics:
   - Fill rate
   - Average confidence
   - Chunks scanned
5. Determine if retry is needed
```

### 7. Finalization Layer

**Responsibility**: Compute metrics and assemble output

**Metrics Computed**:
- Field fill rate (%)
- Retry lift (% improvement)
- Average confidence score
- Processing time per agent
- Total chunks scanned
- Error count

**Output Format**:
```json
{
  "document_id": "...",
  "fields": {
    "field_name": {
      "value": "...",
      "confidence": 0.92,
      "citation": {
        "source": "attachment",
        "attachment_name": "doc.pdf",
        "chunk_id": "uuid",
        "page": 3,
        "line_numbers": [15, 16],
        "extraction_pass": "initial"
      }
    }
  },
  "metrics": {...},
  "errors": [...]
}
```

## LangGraph Implementation

### Graph Structure

```
StateGraph(GraphState)
  ├─ preprocess (START)
  ├─ parse_documents
  ├─ chunk_content
  ├─ parallel_extract
  ├─ aggregate
  ├─ [conditional] → retry_extraction OR finalize
  └─ finalize (END)
```

### State Management

**GraphState** carries all data through the workflow:
- Input: email_path, document_id, field_metadata
- Intermediate: parsed_contents, chunks, agent_states
- Output: extracted_fields, metrics, errors

**State Updates**:
- Each node receives current state
- Node modifies relevant fields
- Returns updated state
- LangGraph handles state propagation

### Conditional Edges

```python
def should_retry(state: GraphState) -> str:
    has_missing = len(state.missing_fields) > 0
    under_limit = state.retry_count < state.max_retries
    
    if has_missing and under_limit:
        return "retry"
    return "finalize"
```

## Data Flow

```
Email File (.eml/.msg)
  ↓
[EmailParser] → email_headers, email_body, attachments[]
  ↓
[DocumentParser] → parsed_contents[] (normalized text)
  ↓
[TextChunker] → chunks[] (with provenance)
  ↓
[5x CategoryAgent] → agent_states{} (parallel)
  ↓
[Aggregator] → extracted_fields{}, missing_fields[]
  ↓
[Decision] → Retry? Yes/No
  ↓
[Retry] → Re-extract missing fields
  ↓
[Finalize] → ExtractionResult with metrics
```

## Error Handling Strategy

### Levels of Isolation

1. **Node-Level**: Try-catch in each node, log and continue
2. **Agent-Level**: Exception isolation per agent
3. **Field-Level**: Missing field doesn't fail extraction
4. **System-Level**: Graceful degradation with partial results

### Error Propagation

```python
ErrorRecord = {
    "error_type": "ParseError",
    "component": "parse_documents",
    "recoverable": True,
    "context": {...}
}

# Accumulated in state.errors
# Doesn't stop pipeline
# Reported in final output
```

## Scalability Patterns

### 1. Category Partitioning
- Split field categories across worker nodes
- Each worker runs subset of agents
- Results merged via message queue

### 2. Document Batching
- Process multiple emails in parallel
- Shared chunk cache across documents
- Batch LLM calls for same fields

### 3. Caching Strategy
- **L1**: In-memory chunk cache
- **L2**: Redis for parsed documents
- **L3**: S3/Blob storage for attachments

### 4. Load Balancing
```
API Gateway
  ├─ Worker Pool 1 (Company + Financial)
  ├─ Worker Pool 2 (Contract + Compliance)
  └─ Worker Pool 3 (Personnel)
```

## Security Architecture

### Data Protection
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII redaction in logs
- Field-level encryption for sensitive data

### Access Control
```
Roles:
  - Admin: Full access
  - Operator: Run extractions
  - Viewer: Read results only
  - API: Programmatic access with keys
```

### Audit Trail
- All extractions logged with timestamp
- User attribution
- Data lineage tracking
- Immutable audit logs

## Future Extensions

### 1. Vector Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed chunks
embeddings = model.encode([chunk.content for chunk in chunks])

# Semantic search for relevant chunks
relevant_chunks = semantic_search(field_description, embeddings)
```

### 2. Active Learning
- Track low-confidence extractions
- Request human validation
- Retrain extraction prompts
- Build field-specific examples

### 3. Multi-Language Support
```python
from langdetect import detect

language = detect(email_body)
prompts = load_prompts(language)
```

### 4. Real-Time Streaming
```python
async def stream_extraction(email_path: str):
    async for field in extract_stream(email_path):
        yield {
            "field": field.name,
            "value": field.value,
            "progress": calculate_progress()
        }
```

### 5. Human-in-the-Loop
```python
if field.confidence < 0.6:
    # Request human validation
    validated = await request_human_review(field)
    field.value = validated.value
    field.confidence = 1.0
```

## Performance Characteristics

### Time Complexity
- Email parsing: O(n) where n = email size
- Document parsing: O(m) where m = attachment size
- Chunking: O(t) where t = total text length
- Extraction per field: O(c) where c = chunk count
- Total: O(n + m + t + f*c) where f = field count

### Space Complexity
- Chunks in memory: O(total_text_length)
- Agent states: O(num_agents * num_fields)
- Can stream-process to reduce memory

### Bottlenecks
1. **LLM API calls**: Rate limited, latency ~1-2s per call
2. **Document parsing**: OCR can be slow (~5s per page)
3. **Memory**: Large attachments can consume significant RAM

### Optimizations Applied
- Parallel agent execution (3-5x speedup)
- Async I/O (non-blocking)
- Early stopping on confident match
- Chunk-level caching
- Retry only missing fields

## Monitoring Dashboard

Recommended metrics to track:

```
Extraction Metrics:
  - Fill rate (%)
  - Avg confidence
  - Processing time
  - Retry lift

System Metrics:
  - Memory usage
  - CPU utilization
  - API call latency
  - Error rate

Business Metrics:
  - Extractions per hour
  - Cost per extraction
  - SLA compliance
  - User satisfaction
```
