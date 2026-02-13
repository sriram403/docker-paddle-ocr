# JLR Doc Intelligence API
GPU-accelerated Document Intelligence service built with **FastAPI**, **PaddleOCR**, and **Docker**.

This service processes PDF documents via an API endpoint, extracts structured clauses, applies AI labeling and summaries, and stores results as JSON files following a strict output naming convention.

---

## 🚀 Overview

The JLR Doc Intelligence API is a containerized microservice designed for regulatory document analysis.

Main capabilities:

- Download PDF from URL
- OCR + Layout Analysis using PaddleOCR
- Clause detection & hierarchy reconstruction
- AI table extraction (optional)
- AI bundle analysis & summary generation
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

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e OUTPUT_DIR=/outputs \
  -e OPENAI_API_KEY=YOUR_KEY \
  -v $(pwd)/outputs:/outputs \
  -v ~/paddle_models:/root/.paddleocr \
  jlr-doc-intel
```

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
  "enable_ai_tables": true,
  "do_summary": true
}
```

#### Example Python Client

```python
import requests

url = "http://localhost:8000/doc-intel"

payload = {
    "document_url": "...",
    "request_id": "test123",
    "num_pages": 5,
    "model_type": "v3",
    "enable_ai_tables": True,
    "do_summary": True
}

response = requests.post(url, json=payload)
print(response.json())
```
Locally

```python
import requests
import json
from datetime import datetime

url = "http://AWS_Public_IP:8000/doc-intel"

payload = {
    "document_url": "https://drive.google.com/uc?export=download&id=1zffFZhQeN_Q8AheoPKePZWZOr5YLnC7j",
    "request_id": "test123",
    "num_pages": 5,
    "model_type": "v3",
    "enable_ai_tables": True,
    "do_summary": True
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    req_id = payload.get("request_id", "NA")

    filename = f"LOCAL__DOCINTEL__{ts}__{req_id}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved locally as {filename}, also created on the server.")
```

---

## 📄 Output JSON

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

Top-level keys:

- request_id
- job_id
- created_at
- status
- output_file
- data

---

## 🔧 Environment Variables

| Variable | Description | Default |
|---|---|---|
| OUTPUT_DIR | Output folder inside container | /outputs |
| CLIENT_NAME | Filename prefix | JLR |
| OCR_ENGINE | OCR backend | paddleocr |
| OPENAI_API_KEY | OpenAI API key | "" |
| REQUEST_TIMEOUT_S | Download timeout | 60 |
| MAX_DOWNLOAD_MB | Max file size | 100 |

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

```
curl http://localhost:8000/health
```

---

## ⚠️ Notes

- Files written to `/outputs` exist **inside the container** unless mounted with Docker volumes.
- To access outputs locally:

```bash
-v ./outputs:/outputs
```

- API responses always contain the full JSON result even without volume mounts.

---

## 📜 License

Internal JLR Development Build.
