# Deployment & Scalability Guide

## Production Deployment

### Environment Setup

1. **Environment Variables**
```bash
export OPENAI_API_KEY="your-api-key"
export PARSER_API_KEY="your-parser-key"
export PARSER_ENDPOINT="https://your-parser.com/api"
export LOG_LEVEL="INFO"
```

2. **Configuration**
```bash
cp config/extraction_config.yaml config/production.yaml
# Edit production.yaml with production settings
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t email-extraction:latest .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY email-extraction:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: email-extraction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: email-extraction
  template:
    metadata:
      labels:
        app: email-extraction
    spec:
      containers:
      - name: extractor
        image: email-extraction:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

## Scalability Considerations

### Horizontal Scaling

**1. Worker Pool Pattern**
```python
from concurrent.futures import ProcessPoolExecutor

async def process_batch(email_paths: List[str]):
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(extract_from_email, path)
            for path in email_paths
        ]
        results = [f.result() for f in futures]
    return results
```

**2. Message Queue Integration**
```python
# Using Celery for distributed task processing
from celery import Celery

app = Celery('extraction', broker='redis://localhost:6379')

@app.task
def extract_task(email_path: str, document_id: str):
    result = asyncio.run(extract_from_email(email_path, document_id))
    return result

# Submit tasks
for email in emails:
    extract_task.delay(email['path'], email['id'])
```

**3. Kafka/RabbitMQ Integration**
```python
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('extraction-requests')
producer = KafkaProducer('extraction-results')

for message in consumer:
    request = json.loads(message.value)
    result = await extract_from_email(**request)
    producer.send('extraction-results', json.dumps(result))
```

### Caching Strategy

**1. Document Cache (Redis)**
```python
import redis
import pickle

cache = redis.Redis(host='localhost', port=6379)

async def get_parsed_document(attachment_path: str):
    cached = cache.get(f"parsed:{attachment_path}")
    if cached:
        return pickle.loads(cached)
    
    parsed = await parse_document(attachment_path)
    cache.setex(f"parsed:{attachment_path}", 3600, pickle.dumps(parsed))
    return parsed
```

**2. Chunk Cache**
```python
async def get_chunks(content_hash: str):
    cached = cache.get(f"chunks:{content_hash}")
    if cached:
        return pickle.loads(cached)
    
    chunks = create_chunks(content)
    cache.setex(f"chunks:{content_hash}", 7200, pickle.dumps(chunks))
    return chunks
```

### Database Integration

**PostgreSQL Schema for Tracking**
```sql
CREATE TABLE extraction_jobs (
    id UUID PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    fill_rate DECIMAL(5,2),
    error_count INTEGER
);

CREATE TABLE extracted_fields (
    id UUID PRIMARY KEY,
    job_id UUID REFERENCES extraction_jobs(id),
    field_name VARCHAR(255),
    field_value TEXT,
    confidence DECIMAL(5,2),
    source VARCHAR(50),
    attachment_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_job_status ON extraction_jobs(status);
CREATE INDEX idx_field_job ON extracted_fields(job_id);
```

### Performance Optimization

**1. Parallel Agent Execution**
```python
# Already implemented in AgentOrchestrator
async def run_all_agents(self, chunks):
    tasks = [agent.extract(chunks) for agent in self.agents.values()]
    results = await asyncio.gather(*tasks)
    return results
```

**2. Batch Processing**
```python
async def process_email_batch(email_paths: List[str], batch_size: int = 10):
    """Process emails in batches to avoid overwhelming resources."""
    results = []
    for i in range(0, len(email_paths), batch_size):
        batch = email_paths[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            extract_from_email(path, f"DOC-{i+j}")
            for j, path in enumerate(batch)
        ])
        results.extend(batch_results)
        await asyncio.sleep(1)  # Rate limiting
    return results
```

**3. Streaming Processing**
```python
async def stream_process_large_attachment(attachment_path: str):
    """Stream process large attachments to avoid memory issues."""
    async with aiofiles.open(attachment_path, 'rb') as f:
        async for chunk in read_in_chunks(f, chunk_size=1024*1024):
            # Process chunk
            yield process_chunk(chunk)
```

### Monitoring & Observability

**1. Prometheus Metrics**
```python
from prometheus_client import Counter, Histogram, Gauge

extraction_counter = Counter('extraction_total', 'Total extractions')
extraction_duration = Histogram('extraction_duration_seconds', 'Extraction duration')
active_extractions = Gauge('extraction_active', 'Active extractions')

@extraction_duration.time()
async def extract_with_metrics(email_path: str, document_id: str):
    extraction_counter.inc()
    active_extractions.inc()
    try:
        result = await extract_from_email(email_path, document_id)
        return result
    finally:
        active_extractions.dec()
```

**2. Distributed Tracing (OpenTelemetry)**
```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer(__name__)

async def extract_from_email(email_path: str, document_id: str):
    with tracer.start_as_current_span(
        "extract_email",
        kind=SpanKind.SERVER,
        attributes={"document_id": document_id}
    ) as span:
        # Extraction logic
        span.set_attribute("fields_extracted", len(result.fields))
        return result
```

**3. Health Checks**
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "llm": await check_llm_health(),
            "parser": await check_parser_health(),
            "cache": await check_cache_health()
        }
    }
```

## Load Testing

```python
# locustfile.py
from locust import User, task, between
import asyncio

class ExtractionUser(User):
    wait_time = between(1, 3)
    
    @task
    def extract_email(self):
        asyncio.run(extract_from_email(
            "test_email.eml",
            f"DOC-{self.user_id}"
        ))
```

Run with:
```bash
locust -f locustfile.py --users 100 --spawn-rate 10
```

## Cost Optimization

1. **Token Usage Tracking**
   - Monitor LLM token consumption per extraction
   - Implement token budgets per field category
   - Cache frequently extracted patterns

2. **Selective Retry**
   - Only retry high-priority fields
   - Use cheaper models for retry attempts
   - Implement confidence thresholds to skip retries

3. **Batch API Calls**
   - Combine multiple field extractions in single LLM call
   - Use batch processing for document parsing

## Security Considerations

1. **Data Encryption**
   - Encrypt sensitive fields in database
   - Use TLS for all API communication
   - Implement data retention policies

2. **Access Control**
   - Role-based access control (RBAC)
   - API key rotation
   - Audit logging

3. **PII Protection**
   - Redact sensitive information
   - Implement data anonymization
   - GDPR compliance features

## Disaster Recovery

1. **Backup Strategy**
   - Regular database backups
   - Configuration versioning
   - Extraction job state persistence

2. **Failure Recovery**
   - Automatic retry with exponential backoff
   - Dead letter queue for failed extractions
   - Circuit breaker pattern for external APIs

## Performance Benchmarks

Expected performance on standard hardware (4 CPU, 8GB RAM):

| Metric | Value |
|--------|-------|
| Single email extraction | 15-30s |
| Fields per second | 5-10 |
| Concurrent extractions | 10-20 |
| Memory per extraction | 500MB-1GB |
| Throughput (emails/hour) | 100-200 |

With optimizations:
- Caching: 40% speed improvement
- Parallel agents: 3x throughput
- Batch processing: 5x throughput
