# JLR Doc Intelligence API
GPU-accelerated Document Intelligence service built with **FastAPI**, **PaddleOCR**, and **Docker**.

This service processes PDF documents via an API endpoint, extracts structured clauses, applies AI labeling and summaries, and stores results as JSON files following a strict output naming convention.

**NEW:** Now supports both **OpenAI GPT** and **Vertex AI Gemini Pro** for AI-powered analysis with runtime provider selection.

---

## 🚀 Overview

The JLR Doc Intelligence API is a containerized microservice designed for regulatory document analysis.

Main capabilities:

- Download PDF from URL
- OCR + Layout Analysis using PaddleOCR
- Clause detection & hierarchy reconstruction
- **AI table extraction** (OpenAI GPT-4o or Vertex AI Gemini)
- **AI bundle analysis & summary generation** (OpenAI or Vertex AI)
- **Runtime AI provider selection** (choose per request or via env var)
- Structured JSON output
- Automatic output file storage

The service runs inside Docker and exposes a REST API.

---

## 🧱 Architecture

```
Client Script
      │
      ▼
 FastAPI (app.main)
      │
      ▼
DocIntelService
      │
      ▼
DocumentProcessor (PaddleOCR + AI)
      │
      ▼
Output Writer → JSON Files
```

---

## 📂 Project Structure

```
app/
 ├── main.py                 # FastAPI app entrypoint
 ├── core/
 │    ├── config.py
 │    ├── logger.py
 │    ├── ai_client.py       # NEW: AI provider abstraction (OpenAI/Vertex)
 │    └── model_registry.py
 ├── routes/
 │    ├── doc_intel.py
 │    └── health.py
 ├── services/
 │    └── doc_intel_service.py
 ├── engine/
 │    └── doc_processor_engine.py
 └── storage/
      └── output_writer.py

Dockerfile
vertex_smoke_test.py        # NEW: Vertex AI validation script
```

---

## ⚙️ Requirements

- Docker
- NVIDIA GPU + CUDA (for GPU mode)
- NVIDIA Container Toolkit

---

## 🐳 Docker Setup

### 1. Build Image

```bash
docker build -t jlr-doc-intel .
```

### 2. Run Container

#### Option A: Using OpenAI (AWS, Local Development)

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e AI_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e OUTPUT_DIR=/outputs \
  -v $(pwd)/outputs:/outputs \
  -v ~/paddle_models:/root/.paddleocr \
  jlr-doc-intel
```

#### Option B: Using Vertex AI (GCP Production)

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e AI_PROVIDER=vertex \
  -e GCP_PROJECT_ID=jlr-dl-iqm \
  -e GCP_LOCATION=europe-west2 \
  -e OUTPUT_DIR=/outputs \
  -v $(pwd)/outputs:/outputs \
  -v ~/paddle_models:/root/.paddleocr \
  jlr-doc-intel
```

**Note:** Vertex AI uses Application Default Credentials (ADC) - no API key needed. Ensure the VM has a service account with `roles/aiplatform.user`.

---

## 🌐 API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{ "status": "ok" }
```

---

### Document Intelligence

```
POST /doc-intel
```

#### Request Body

```json
{
  "document_url": "PDF_URL",
  "request_id": "optional_id",
  "num_pages": 5,
  "model_type": "v3",
  "ai_provider": "openai",
  "analysis_model": "gpt-4o-mini",
  "enable_ai_tables": true,
  "do_summary": true
}
```

**New Parameters:**
- `ai_provider` (optional): `"openai"` or `"vertex"` - defaults to env var `AI_PROVIDER`
- `analysis_model` (optional): Model name for summaries (e.g., `"gpt-4o-mini"` or `"gemini-2.0-flash-exp"`)

#### Example Python Client

**Using OpenAI:**
```python
import requests

url = "http://localhost:8000/doc-intel"

payload = {
    "document_url": "https://example.com/document.pdf",
    "request_id": "test123",
    "num_pages": 5,
    "model_type": "v3",
    "ai_provider": "openai",
    "analysis_model": "gpt-4o-mini",
    "enable_ai_tables": True,
    "do_summary": True
}

response = requests.post(url, json=payload)
print(response.json())
```

**Using Vertex AI:**
```python
import requests

url = "http://GCP_VM_IP:8000/doc-intel"

payload = {
    "document_url": "https://example.com/document.pdf",
    "request_id": "test456",
    "num_pages": 5,
    "model_type": "v3",
    "ai_provider": "vertex",
    "analysis_model": "gemini-2.0-flash-exp",
    "enable_ai_tables": True,
    "do_summary": True
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 📄 Output JSON Schema

### Pub/Sub Compliant Event Format

The API now returns responses that comply with the **FileProcessingCompleted** event schema for Pub/Sub integration.

Every request:

1. Returns JSON in API response
2. Writes a copy to `OUTPUT_DIR`

Filename rule:

```
{CLIENT}__{FEATURE}__{TIMESTAMP}__{JOB_ID}__{REQUEST_ID}.json
```

Example:

```
JLR__DOCINTEL__20260212T081913__job_xxxx__test123.json
```

### Response Schema

**Success Response:**

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "FileProcessingCompleted",
  "timestamp": "2026-03-16T10:30:00.123456Z",
  "source": "RMS AI",
  "payload": {
    "regulation_change_id": 12345,
    "document_version": 2,
    "processed_bucket_path": "/outputs/JLR__DOCINTEL__20260316T103000__job_xxx__test123.json",
    "processing_status": "SUCCESS",
    "processing_time_ms": 45230,
    "error_message": null,
    "summary_metadata": {
      "pages_processed": 10,
      "confidence_score": 0.98,
      "total_chunks_processed": 156,
      "unique_labels_detected": 8,
      "ai_provider_used": "openai",
      "model_type": "v3",
      "analysis_model": "gpt-4o-mini",
      "tables_detected": 5,
      "summary_generated": true
    },
    "data": {
      "results": [...],
      "metrics": {...},
      "summary": {...}
    }
  },
  "request_id": "test123",
  "job_id": "job_20260316_103000_a1b2",
  "output_file": "/outputs/JLR__DOCINTEL__20260316T103000__job_xxx__test123.json"
}
```

**Error Response:**

```json
{
  "event_id": "...",
  "event_type": "FileProcessingCompleted",
  "timestamp": "2026-03-16T10:30:00Z",
  "source": "RMS AI",
  "payload": {
    "regulation_change_id": null,
    "document_version": null,
    "processed_bucket_path": null,
    "processing_status": "FAILURE",
    "processing_time_ms": 1250,
    "error_message": "PDF download failed: Connection timeout",
    "summary_metadata": {
      "pages_processed": 0,
      "confidence_score": 0.0,
      "total_chunks_processed": 0,
      "unique_labels_detected": 0,
      "ai_provider_used": "openai",
      "model_type": "v3"
    }
  }
}
```

### Field Descriptions

**Top-level Fields:**
- `event_id` (string, UUID): Unique identifier for this processing event
- `event_type` (string): Always "FileProcessingCompleted"
- `timestamp` (string, ISO-8601): UTC timestamp when processing completed
- `source` (string): Always "RMS AI"
- `request_id` (string, optional): Client-provided request identifier
- `job_id` (string): System-generated job identifier
- `output_file` (string): Full path to the saved JSON output file

**Payload Fields:**
- `regulation_change_id` (number, nullable): Regulation change identifier from RMS
- `document_version` (number, nullable): Document version number
- `processed_bucket_path` (string): GCS path or local path to output file
- `processing_status` (string): "SUCCESS" or "FAILURE"
- `processing_time_ms` (number): Total processing time in milliseconds
- `error_message` (string, nullable): Error description if status is FAILURE

**Summary Metadata:**
- `pages_processed` (number): Number of pages analyzed
- `confidence_score` (number): OCR quality metric (0.0 to 1.0)
- `total_chunks_processed` (number): Number of clauses/chunks extracted
- `unique_labels_detected` (number): Count of different regulatory labels found
- `ai_provider_used` (string): "openai" or "vertex"
- `model_type` (string): PaddleOCR model type ("v3" or "vl")
- `analysis_model` (string): AI model used for analysis
- `tables_detected` (number): Number of tables found in document
- `summary_generated` (boolean): Whether AI summary was created

**Data Object:**
- `results` (array): Full structured document chunks with metadata
- `metrics` (object): OCR metrics (word counts, detected tables)
- `summary` (object): AI-generated executive summary by category

### Optional Request Parameters for Pub/Sub Integration

Include these fields in your API request for full Pub/Sub compliance:

```json
{
  "document_url": "https://example.com/doc.pdf",
  "regulation_change_id": 12345,
  "document_version": 2,
  "gcs_file_path": "gs://bucket/regulations/doc.pdf",
  "request_id": "rms-req-001",
  "num_pages": 10,
  "ai_provider": "openai",
  "enable_ai_tables": true,
  "do_summary": true
}
```

---

## 🔧 Environment Variables

### Core Configuration
| Variable | Description | Default |
|---|---|---|
| OUTPUT_DIR | Output folder inside container | /outputs |
| CLIENT_NAME | Filename prefix | JLR |
| OCR_ENGINE | OCR backend | paddleocr |
| REQUEST_TIMEOUT_S | Download timeout (seconds) | 60 |
| MAX_DOWNLOAD_MB | Max file size (MB) | 100 |

### AI Provider Configuration
| Variable | Description | Default | Required For |
|---|---|---|---|
| **AI_PROVIDER** | AI provider choice | `openai` | Both |
| **OPENAI_API_KEY** | OpenAI API key | `""` | OpenAI mode |
| **GCP_PROJECT_ID** | GCP project ID | `""` | Vertex AI mode |
| **GCP_LOCATION** | Vertex AI region | `europe-west2` | Vertex AI mode |
| **VERTEX_MODEL** | Gemini model name | `gemini-2.0-flash-exp` | Vertex AI mode |

**Provider Options:**
- `AI_PROVIDER=openai` → Uses OpenAI GPT-4 (requires `OPENAI_API_KEY`)
- `AI_PROVIDER=vertex` → Uses Vertex AI Gemini (requires `GCP_PROJECT_ID` and ADC)

---

## 🧠 Model Loading

At startup the API:

- Creates output directory
- Preloads PaddleOCR model into GPU memory
- Keeps model cached to avoid reload per request

---

## 📊 Features

- GPU-accelerated OCR
- Layout-aware document parsing
- Table detection + AI extraction
- Regulatory clause labeling
- Executive summary generation
- Atomic JSON output writing

---

## 🛠 Development

Run locally without Docker:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Vertex AI Smoke Test (GCP Only)

Before deploying to production, validate Vertex AI access:

```bash
# 1. Create virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install google-genai>=0.5.0 google-auth>=2.30.0

# 2. Set environment variables
export PROJECT_ID="jlr-dl-iqm"
export LOCATION="europe-west2"
export MODEL="gemini-2.0-flash-exp"

# 3. Run smoke test
python vertex_smoke_test.py "Reply with: Vertex smoke test OK"
```

**Expected output:**
```
[auth] Using ADC. adc_project='jlr-dl-iqm' target_project='jlr-dl-iqm'
[request] model=gemini-2.0-flash-exp location=europe-west2

=== MODEL RESPONSE ===
Vertex smoke test OK. I'm running in europe-west2.

✅ Smoke test PASSED
```

---

## 🔐 GCP Deployment (Vertex AI)

### Prerequisites

1. **Enable Vertex AI API:**
   ```bash
   gcloud services enable aiplatform.googleapis.com --project=jlr-dl-iqm
   ```

2. **Grant Service Account Permissions:**
   ```bash
   gcloud projects add-iam-policy-binding jlr-dl-iqm \
     --member="serviceAccount:YOUR_SA@jlr-dl-iqm.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Verify ADC on VM:**
   ```bash
   curl -H "Metadata-Flavor: Google" \
     http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email
   ```

### Troubleshooting

**403 PERMISSION_DENIED:**
- The VM's service account is missing `roles/aiplatform.user`
- Check IAM permissions in GCP Console

**404 NOT_FOUND or model not recognized:**
- Model name might not be available in your region
- Try `gemini-1.5-pro` or `gemini-1.5-flash` instead

**429 RESOURCE_EXHAUSTED:**
- Vertex AI quota limit reached
- Check quotas: GCP Console → IAM & Admin → Quotas

**Module not found errors:**
- Rebuild Docker image to include new dependencies
- Verify `google-genai` and `google-auth` are in requirements

---

## 🎯 Complete Usage Examples

### Example 1: Quick Test with OpenAI (Local/AWS)

```bash
# 1. Build the image
docker build -t jlr-doc-intel .

# 2. Run container
docker run --rm --gpus all \
  -p 8000:8000 \
  -e AI_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-your-actual-key \
  -v $(pwd)/outputs:/outputs \
  jlr-doc-intel

# 3. Test health endpoint (in another terminal)
curl http://localhost:8000/health

# 4. Test document processing
curl -X POST http://localhost:8000/doc-intel \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "https://example.com/your-document.pdf",
    "request_id": "test-001",
    "num_pages": 5,
    "ai_provider": "openai",
    "enable_ai_tables": true,
    "do_summary": true
  }'
```

### Example 2: Python Client Script

Create a file `test_api.py`:

```python
import requests
import json

# Configuration
API_URL = "http://localhost:8000/doc-intel"
PDF_URL = "https://your-pdf-url.com/document.pdf"

# Request payload
payload = {
    "document_url": PDF_URL,
    "request_id": "python-test-001",
    "num_pages": 10,
    "model_type": "v3",
    "ai_provider": "openai",  # or "vertex" for GCP
    "analysis_model": "gpt-4o-mini",
    "enable_ai_tables": True,
    "do_summary": True
}

# Send request
print(f"📤 Sending request to {API_URL}")
response = requests.post(API_URL, json=payload)

# Check response
if response.status_code == 200:
    result = response.json()
    print(f"✅ Status: {result.get('status')}")
    print(f"📄 Output file: {result.get('output_file')}")

    # Save response locally
    with open(f"response_{payload['request_id']}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Saved response locally")
else:
    print(f"❌ Error {response.status_code}: {response.text}")
```

Run it:
```bash
pip install requests
python test_api.py
```

### Example 3: Switching Between Providers

**Same container, different requests:**

```python
import requests

# Request 1: Use OpenAI
requests.post("http://localhost:8000/doc-intel", json={
    "document_url": "https://example.com/doc1.pdf",
    "ai_provider": "openai",
    "analysis_model": "gpt-4o-mini",
    "num_pages": 5
})

# Request 2: Use Vertex AI (if on GCP)
requests.post("http://localhost:8000/doc-intel", json={
    "document_url": "https://example.com/doc2.pdf",
    "ai_provider": "vertex",
    "analysis_model": "gemini-2.0-flash-exp",
    "num_pages": 5
})
```

---

## 📋 Step-by-Step Testing Guide

### Phase 1: Local Testing (OpenAI)

**Step 1: Stop existing containers**
```bash
docker ps -a
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
```

**Step 2: Build fresh image**
```bash
cd /home/ubuntu/Sriram/paddleOCR-Docker
docker build -t jlr-doc-intel .
```

**Step 3: Verify environment**
```bash
# Check if OPENAI_API_KEY is set
echo $OPENAI_API_KEY

# If not set, export it
export OPENAI_API_KEY="sk-your-key-here"
```

**Step 4: Run container**
```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e AI_PROVIDER=openai \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/outputs:/outputs \
  --name jlr-doc-intel-test \
  jlr-doc-intel
```

**Step 5: Test (in new terminal)**
```bash
# Health check
curl http://localhost:8000/health

# Process a document
curl -X POST http://localhost:8000/doc-intel \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "YOUR_PDF_URL",
    "request_id": "test-openai-001",
    "num_pages": 3,
    "ai_provider": "openai",
    "enable_ai_tables": false,
    "do_summary": false
  }'
```

**Step 6: Check logs**
```bash
docker logs jlr-doc-intel-test
```

**Step 7: Check outputs**
```bash
ls -lh outputs/
```

### Phase 2: GCP Testing (Vertex AI)

**Only run on GCP VM with proper service account:**

**Step 1: Run smoke test**
```bash
cd /home/ubuntu/Sriram/paddleOCR-Docker
export PROJECT_ID="jlr-dl-iqm"
export LOCATION="europe-west2"
python vertex_smoke_test.py "Test message"
```

**Step 2: If smoke test passes, run container**
```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e AI_PROVIDER=vertex \
  -e GCP_PROJECT_ID=jlr-dl-iqm \
  -e GCP_LOCATION=europe-west2 \
  -v $(pwd)/outputs:/outputs \
  jlr-doc-intel
```

**Step 3: Test with Vertex AI**
```bash
curl -X POST http://localhost:8000/doc-intel \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "YOUR_PDF_URL",
    "request_id": "test-vertex-001",
    "num_pages": 3,
    "ai_provider": "vertex",
    "analysis_model": "gemini-2.0-flash-exp",
    "enable_ai_tables": true,
    "do_summary": true
  }'
```

---

## ⚠️ Important Notes

### General
- **Backward Compatibility:** Existing API calls without `ai_provider` will use OpenAI (default)
- **No Restart Needed:** Provider can be changed per request without restarting container
- **Model Names Matter:**
  - OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
  - Vertex AI: `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`

### OpenAI Mode (AWS/Local)
- ✅ Works on any environment (AWS, local, GCP)
- ✅ Requires `OPENAI_API_KEY` environment variable
- ✅ No special permissions needed
- ⚠️ API costs apply per token usage

### Vertex AI Mode (GCP Only)
- ✅ Uses Application Default Credentials (ADC) - no API keys in code
- ✅ More secure authentication method
- ⚠️ **Only works on GCP VMs** with proper service account
- ⚠️ Requires `roles/aiplatform.user` permission
- ⚠️ Requires Vertex AI API enabled in project
- ⚠️ Must run smoke test first to validate setup

### Troubleshooting Common Issues

**Container won't start:**
```bash
# Check if port 8000 is already in use
lsof -i :8000
# Kill the process or use different port
docker run -p 8001:8000 ...
```

**OpenAI authentication fails:**
```bash
# Verify API key is valid
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Vertex AI fails with 403:**
```bash
# Check service account
curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email

# Check permissions
gcloud projects get-iam-policy jlr-dl-iqm \
  --flatten="bindings[].members" \
  --filter="bindings.role:roles/aiplatform.user"
```

**PDF download fails:**
```bash
# Test PDF URL is accessible
curl -I YOUR_PDF_URL

# Check timeout settings (default 60s)
docker run -e REQUEST_TIMEOUT_S=120 ...
```

**GPU not detected:**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check NVIDIA Container Toolkit
nvidia-container-cli --version
```

### Performance Tips

1. **Model Caching:** PaddleOCR models are cached after first load
2. **Batch Processing:** Process multiple pages but start with small `num_pages` for testing
3. **AI Tables:** Set `enable_ai_tables=false` for faster processing if tables not needed
4. **Summaries:** Set `do_summary=false` to skip AI analysis and speed up processing

### Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **On GCP, prefer ADC** over API keys
4. **Rotate API keys** regularly
5. **Monitor API usage** to detect anomalies
6. **Use HTTPS** in production (not HTTP)

---

## ⚠️ Additional Notes

- Files written to `/outputs` exist **inside the container** unless mounted with Docker volumes.
- To access outputs locally use: `-v $(pwd)/outputs:/outputs`
- API responses always contain the full JSON result even without volume mounts.
- Check `outputs/` folder for saved JSON files with structured results

---

## 📜 License

Internal JLR Development Build.
