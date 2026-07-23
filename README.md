# JLR Document Intelligence API

GPU-accelerated regulatory-document extraction based on the current
`experimentation` pipeline. The production interface remains FastAPI with
optional Google Cloud Pub/Sub input and GCS output.

The service extracts document hierarchy, clauses, tables, figures, references,
topic labels, requirement labels, text-type labels, and an optional regulatory
briefing.

> Customer support shortcut: if the model container repeatedly restarts, send
> the customer directly to [Known deployment issue: vLLM restart loop](#known-deployment-issue-vllm-restart-loop).

## Runtime architecture

The deployment uses two GPU-enabled containers on one private Compose network.
Only FastAPI is published to the host; vLLM is internal.

```text
                              Docker host
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Client ──HTTP :8000──► api container                               │
│                        ├─ FastAPI routes                             │
│                        ├─ DocIntelService                            │
│                        ├─ isolated Paddle layout worker ──GPU        │
│                        └─ outputs / logs / checkpoints ──host mounts │
│                                      │                              │
│                                      │ OpenAI-compatible /v1        │
│                                      ▼                              │
│                        vllm container ── PaddleOCR-VL-1.5 ──GPU      │
│                        internal port 8080 (not host-published)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

- `api` exposes port 8000, listens for Pub/Sub events, and runs the isolated
  Paddle layout worker.
- `vllm` serves `PaddleOCR-VL-1.5-0.9B` internally on port 8080.
- Jobs are serialized inside the API process to protect shared GPU memory.
- Pub/Sub flow control also limits the subscriber to one outstanding message.

### HTTP request flow

```text
POST /doc-intel
      │
      ▼
Validate request and download the PDF
      │
      ▼
DocIntelService
      │
      ├─ Paddle worker: layout detection, OCR and structure
      ├─ vLLM: PaddleOCR-VL recognition
      └─ Optional OpenAI / Gemini / Vertex analysis
      │
      ▼
Atomic JSON write to ./outputs
      │
      ▼
Full FileProcessingCompleted-shaped response returned to the client
```

### Pub/Sub automated flow

```text
RMS publishes PDFUploaded
      │
      ▼
PUBSUB_TOPIC
      │
      ▼
PUBSUB_SUBSCRIPTION
      │
      ▼
API background listener (maximum one outstanding message)
      │
      ├─ publish FileProcessingStarted
      ├─ download source PDF from GCS
      ├─ run the same DocIntelService pipeline
      ├─ save full JSON locally
      ├─ upload full JSON to gs://<bucket>/<input-dir>/output/<file>.json
      └─ publish metadata-only FileProcessingCompleted
      │
      ▼
ack on completion or permanent bad input; nack transient GCS download failure
```

### Repository layout

```text
app/
├── main.py                         FastAPI lifecycle and listener startup
├── core/
│   ├── ai_client.py                OpenAI, Gemini and Vertex abstraction
│   ├── config.py                   Environment configuration
│   ├── logger.py / logging.py      Application and document logging
│   └── model_registry.py           Lazy Paddle worker registration
├── engine/
│   ├── doc_processor_engine.py     Current extraction pipeline
│   ├── paddle_worker_process.py    Isolated Paddle subprocess
│   └── progress.py                 Headless progress compatibility
├── routes/
│   ├── doc_intel.py                POST /doc-intel
│   ├── health.py                   GET /health and /ready
│   └── pubsub_test.py              GET /test-pubsub in Pub/Sub mode
├── services/
│   ├── doc_intel_service.py        Pipeline-to-event response adapter
│   └── pubsub_listener.py          Pull, GCS and publish workflow
└── storage/
    └── output_writer.py            Atomic local writes and GCS upload

Dockerfile                          API image
Dockerfile.vllm                     vLLM image
docker-compose.yml                  Two-service GPU deployment
requirements-api.txt                Pinned API dependencies
scripts/smoke_test.py               Local one-page end-to-end test
tests/test_service.py               Response-contract unit test
```

## Tested deployment profile

The complete build and one-page OCR smoke test have been verified with:

- Ubuntu 24.04
- Docker Engine 29.6 and Docker Compose 5.3
- NVIDIA Container Toolkit 1.19
- NVIDIA A10G with 24 GB VRAM
- `PaddleOCR-VL-1.5-0.9B`

Allow at least 70 GB of free disk space before the first build. The current API
and vLLM images are approximately 19 GB and 32 GB respectively, and Docker also
needs temporary space while downloading and unpacking layers. The default GPU
settings are intended for a 24 GB GPU. Smaller GPUs have not been validated.

## Fresh GPU-instance setup

### 1. Install the host GPU runtime

Install a working NVIDIA driver first and confirm that the host can see the
GPU:

```bash
nvidia-smi
```

Install Docker Engine and the Compose plugin using the
[official Docker Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
Then install and configure the NVIDIA Container Toolkit using the
[official NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If Docker was newly installed, add the login user to the Docker group and log
out and back in before continuing:

```bash
sudo usermod -aG docker "$USER"
```

Verify Docker and GPU passthrough:

```bash
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
docker compose version
```

Do not continue until the CUDA container displays the host GPU.

### 2. Download the PaddleOCR-VL model

The vLLM service deliberately uses a pre-downloaded model. Download the
official `PaddlePaddle/PaddleOCR-VL-1.5` Hugging Face repository into the model
cache before starting Compose:

```bash
sudo apt-get install -y python3-venv
python3 -m venv /tmp/paddleocr-hf-download
source /tmp/paddleocr-hf-download/bin/activate
python -m pip install --upgrade huggingface_hub

export PADDLEX_MODEL_ROOT="$(pwd)/models"
mkdir -p "$PADDLEX_MODEL_ROOT"
hf download PaddlePaddle/PaddleOCR-VL-1.5 \
  --local-dir "$PADDLEX_MODEL_ROOT/PaddleOCR-VL-1.5"

deactivate
```

Confirm that the weights and model code exist:

```bash
test -f "$PADDLEX_MODEL_ROOT/PaddleOCR-VL-1.5/model.safetensors"
test -f "$PADDLEX_MODEL_ROOT/PaddleOCR-VL-1.5/config.json"
```

The API worker downloads additional Paddle layout models into this cache on
the first OCR request, so the host needs outbound internet access initially and
the directory must remain writable.

### 3. Configure the application

From this repository:

```bash
cp .env.example .env
```

Review `.env`. Its portable default, `PADDLEX_MODEL_ROOT=./models`, points to
the directory created in step 2 regardless of which user runs Docker. Leave
`ENABLE_PUBSUB_LISTENER=false` for HTTP-only deployments. Add credentials only
for the AI provider and optional features being used.

For example, an OpenAI HTTP-only deployment needs:

```bash
AI_PROVIDER=openai
OPENAI_API_KEY=your-key
ENABLE_PUBSUB_LISTENER=false
PADDLEX_MODEL_ROOT=./models
```

An API key is not used when `enable_ai_tables`, `enable_topic_ai`, and
`do_summary` are all false.

For Vertex AI, configure Application Default Credentials so they are available
inside the API container, and set:

```bash
AI_PROVIDER=vertex
GCP_PROJECT_ID=your-project
GCP_LOCATION=europe-west2
```

For the Gemini Developer API:

```bash
AI_PROVIDER=gemini
GEMINI_API_KEY=your-key
ENABLE_PUBSUB_LISTENER=false
```

### 4. Build and start

```bash
docker compose config --quiet
docker compose up --build -d
```

The first build downloads tens of gigabytes and can take a while. Compose
starts the API only after vLLM becomes healthy. Follow startup with:

```bash
docker compose ps
docker compose logs -f vllm api
```

In another shell, check the deployment:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
docker compose ps
```

`/health` verifies the API process. `/ready` additionally verifies vLLM when
`DEFAULT_MODEL_TYPE=vl`.

Both services should show `Up`, vLLM should show `healthy`, `/health` should
return `{"status":"ok"}`, and `/ready` should return `{"status":"ready",...}`.

### 5. Optional local OCR smoke test

This runs one page through the Paddle layout worker and vLLM using a local PDF:

```bash
API_CONTAINER="$(docker compose ps -q api)"
docker cp /absolute/path/to/test.pdf "${API_CONTAINER}:/tmp/test.pdf"
docker compose exec -T api \
  python3 -m scripts.smoke_test /tmp/test.pdf --pages 1 --model-type vl
```

A successful run exits with status zero and reports at least one extracted
chunk.

## API

Process a PDF available over HTTP:

```bash
curl -X POST http://localhost:8000/doc-intel \
  -H 'Content-Type: application/json' \
  -d '{
    "document_url": "https://example.com/regulation.pdf",
    "request_id": "request-123",
    "num_pages": 10,
    "page_start": 0,
    "model_type": "vl",
    "ai_provider": "openai",
    "analysis_model": "gpt-4o-mini",
    "enable_ai_tables": false,
    "enable_topic_ai": false,
    "do_summary": false,
    "enable_checkpoint": true
  }'
```

Important parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `num_pages` | `10` | Process through this one-based PDF page number. |
| `page_start` | `0` | Zero-based first page, useful for page ranges. |
| `model_type` | `DEFAULT_MODEL_TYPE` | `vl` for PaddleOCR-VL or `v3` for PPStructureV3. |
| `ai_provider` | `AI_PROVIDER` | `openai`, `gemini`, or `vertex`. |
| `enable_ai_tables` | `false` | Use the selected AI provider on detected tables. |
| `enable_topic_ai` | `false` | Refine rule-generated labels with the LLM. |
| `do_summary` | `false` | Generate the regulatory briefing. |
| `enable_checkpoint` | `true` | Save intermediate state for long documents. |

Rule-based topic labels and the current structural extraction run even when AI
options are disabled. API keys are required only for the corresponding enabled
AI operations. Vertex uses ADC rather than an API key.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Confirms that the FastAPI process is running. |
| `GET` | `/ready` | Confirms API readiness and, in VL mode, vLLM health. |
| `POST` | `/doc-intel` | Downloads and processes a PDF. |
| `GET` | `/test-pubsub` | Publishes a connectivity-test message; registered only when Pub/Sub mode is enabled. |
| `GET` | `/docs` | Interactive OpenAPI documentation. |

### Complete request fields

| Field | Required | Default | Description |
|---|---:|---|---|
| `document_url` | yes | — | HTTP(S) URL from which the API downloads the PDF. |
| `request_id` | no | generated/omitted | Caller correlation value and output filename suffix. |
| `num_pages` | no | `10` | Exclusive upper page bound expressed as a one-based page count. |
| `page_start` | no | `0` | Zero-based first page; must be lower than `num_pages`. |
| `model_type` | no | `DEFAULT_MODEL_TYPE` | `vl` or `v3`. |
| `ai_provider` | no | `AI_PROVIDER` | `openai`, `gemini`, or `vertex`. |
| `analysis_model` | no | provider default | Model used by enabled AI analysis. |
| `enable_ai_tables` | no | `false` | Send detected tables to the selected AI provider. |
| `table_model` | no | provider default | Optional model override for table analysis. |
| `enable_topic_ai` | no | `false` | Refine rule-derived labels using the provider. |
| `do_summary` | no | `false` | Produce the regulatory briefing. |
| `enable_checkpoint` | no | `true` | Persist intermediate state for recovery. |
| `document_name` | no | URL filename | Friendly document name stored in output metadata. |
| `regulation_change_id` | no | `null` | RMS regulation-change correlation value. |
| `document_version` | no | `null` | RMS document version. |
| `gcs_file_path` | no | `document_url` | Source GCS correlation value in the event payload. |

The HTTP endpoint downloads only HTTP(S) URLs. Pub/Sub mode handles `gs://`
downloads directly.

### Python client example

```python
import json
import requests

payload = {
    "document_url": "https://example.com/regulation.pdf",
    "request_id": "python-test-001",
    "num_pages": 3,
    "page_start": 0,
    "model_type": "vl",
    "ai_provider": "openai",
    "enable_ai_tables": False,
    "enable_topic_ai": False,
    "do_summary": False,
}

response = requests.post(
    "http://localhost:8000/doc-intel",
    json=payload,
    timeout=1800,
)
response.raise_for_status()
result = response.json()
print(result["payload"]["processing_status"])
print(result["output_file"])

with open("response.json", "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
```

### Output and event contract

Every HTTP request returns the full response and atomically writes the same
JSON under `OUTPUT_DIR`. Output filenames follow:

```text
{CLIENT}__{FEATURE}__{TIMESTAMP}__{JOB_ID}__{REQUEST_ID}.json
```

Example:

```text
JLR__DOCINTEL__20260723T103000__job_20260723_103000_a1b2__request-123.json
```

A successful full response has this shape:

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "FileProcessingCompleted",
  "timestamp": "2026-07-23T10:30:00.123456Z",
  "source": "RMS AI",
  "request_id": "request-123",
  "payload": {
    "regulation_change_id": 12345,
    "document_version": 2,
    "processed_bucket_path": "/outputs/JLR__DOCINTEL__...json",
    "processing_status": "SUCCESS",
    "processing_time_ms": 45230,
    "error_message": null,
    "summary_metadata": {
      "pages_processed": 3,
      "confidence_score": 0.98,
      "total_chunks_processed": 42,
      "unique_labels_detected": 8,
      "ai_provider_used": "openai",
      "model_type": "vl",
      "analysis_model": "gpt-4o-mini",
      "tables_detected": 2,
      "summary_generated": false
    },
    "data": {
      "results": [],
      "metrics": {},
      "summary": {}
    }
  },
  "job_id": "job_20260723_103000_a1b2",
  "output_file": "/outputs/JLR__DOCINTEL__...json"
}
```

On processing failure, the API still returns a
`FileProcessingCompleted`-shaped document with
`payload.processing_status="FAILURE"` and `payload.error_message` populated.
Callers must inspect `processing_status`; a pipeline failure can therefore be
represented in a valid HTTP JSON response rather than solely by an HTTP error.

The Pub/Sub `FileProcessingCompleted` event intentionally omits
`payload.data`. Its `processed_bucket_path` points to the uploaded GCS JSON,
which contains the full `results`, `metrics`, and `summary`.

## Pub/Sub mode

Pub/Sub mode runs alongside the HTTP API. At API startup, a daemon pull
listener subscribes to `PDFUploaded` events and uses the same serialized
document-processing service as `/doc-intel`.

### Event sequence

| Step | Event | Direction | Contents |
|---|---|---|---|
| 1 | `PDFUploaded` | RMS → RMS AI | GCS source path and RMS identifiers. |
| 2 | `FileProcessingStarted` | RMS AI → RMS | Acknowledges that processing began. |
| 3 | `FileProcessingCompleted` | RMS AI → RMS | Status and summary metadata; full data remains in GCS. |

### Expected `PDFUploaded` message

```json
{
  "event_id": "upstream-uuid",
  "event_type": "PDFUploaded",
  "timestamp": "2026-07-23T10:00:00Z",
  "source": "RMS",
  "payload": {
    "regulation_change_id": 12345,
    "document_version": 2,
    "gcs_file_path": "gs://your-bucket/regulations/document.pdf",
    "file_size_bytes": 204800,
    "mime_type": "application/pdf"
  }
}
```

The listener reads `regulation_change_id`, `document_version`, and
`gcs_file_path`. Messages whose `event_type` is not `PDFUploaded` are ignored
and acknowledged.

### Published events

`FileProcessingStarted` includes the source identifiers:

```json
{
  "event_id": "generated-uuid",
  "event_type": "FileProcessingStarted",
  "timestamp": "2026-07-23T10:00:01Z",
  "source": "RMS AI",
  "payload": {
    "regulation_change_id": 12345,
    "document_version": 2,
    "gcs_file_path": "gs://your-bucket/regulations/document.pdf"
  }
}
```

`FileProcessingCompleted` contains metadata rather than the large extraction
payload:

```json
{
  "event_id": "generated-uuid",
  "event_type": "FileProcessingCompleted",
  "timestamp": "2026-07-23T10:01:00Z",
  "source": "RMS AI",
  "payload": {
    "regulation_change_id": 12345,
    "document_version": 2,
    "processed_bucket_path": "gs://your-bucket/regulations/output/JLR__DOCINTEL__...json",
    "processing_status": "SUCCESS",
    "processing_time_ms": 59000,
    "error_message": null,
    "summary_metadata": {
      "pages_processed": 10,
      "confidence_score": 0.98,
      "total_chunks_processed": 156,
      "unique_labels_detected": 8,
      "ai_provider_used": "vertex",
      "model_type": "vl",
      "analysis_model": "gemini-2.5-flash",
      "tables_detected": 5,
      "summary_generated": false
    }
  },
  "job_id": "job_20260723_100100_a1b2",
  "output_file": "gs://your-bucket/regulations/output/JLR__DOCINTEL__...json"
}
```

Both outgoing event types also set a Pub/Sub message attribute named
`event_type`. Downstream subscriptions can filter without decoding the body:

```bash
gcloud pubsub subscriptions create rms-completed-sub \
  --topic=iqm_rms_ai \
  --project=YOUR_PROJECT_ID \
  --message-filter='attributes.event_type = "FileProcessingCompleted"'
```

### Listener configuration

Set these values in `.env`:

```text
ENABLE_PUBSUB_LISTENER=true
PUBSUB_SUBSCRIPTION
PUBSUB_TOPIC
PUBSUB_NUM_PAGES
GCP_PROJECT_ID
GCP_LOCATION
```

`PUBSUB_NUM_PAGES=0` processes the full PDF. Set a positive value to cap
automated jobs at that many pages.

### GCP resources and IAM

Create the topic/subscription once per environment. Replace all example values
before running:

```bash
gcloud pubsub topics create iqm_rms_ai \
  --project=YOUR_PROJECT_ID

gcloud pubsub subscriptions create iqm_rms_ai_upload_sub \
  --topic=iqm_rms_ai \
  --project=YOUR_PROJECT_ID
```

The identity used by the API container requires:

| Role | Why |
|---|---|
| `roles/pubsub.subscriber` | Pull `PDFUploaded` messages. |
| `roles/pubsub.publisher` | Publish started/completed events. |
| `roles/storage.objectViewer` | Download source PDFs. |
| `roles/storage.objectCreator` | Upload completed JSON output. |
| `roles/aiplatform.user` | Required only when Vertex AI operations are enabled. |

Example project-level bindings:

```bash
PROJECT_ID=YOUR_PROJECT_ID
SERVICE_ACCOUNT=YOUR_SA@YOUR_PROJECT_ID.iam.gserviceaccount.com

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/pubsub.subscriber"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/pubsub.publisher"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectCreator"
```

Prefer bucket-, topic-, and subscription-level bindings when the deployment
does not require project-wide access.

On a GCE VM, Application Default Credentials normally come from the attached
service account through the metadata server. Outside GCE, explicitly inject
credentials into the API container; Compose does not create or copy credential
files.

### Runtime behavior and acknowledgement rules

For each valid message, the listener:

1. Publishes `FileProcessingStarted`.
2. Downloads `payload.gcs_file_path`.
3. Processes either the full PDF or `PUBSUB_NUM_PAGES`.
4. Atomically saves the full result under `./outputs`.
5. Uploads it to `gs://<bucket>/<input-directory>/output/<filename>.json`.
6. Publishes metadata-only `FileProcessingCompleted`.
7. Acknowledges the input message.

Flow control sets `max_messages=1`, matching the application-wide processing
lock and shared-GPU limits.

Acknowledgement behavior is deliberate:

| Condition | Action | Reason |
|---|---|---|
| Malformed JSON | `ack` | Retrying cannot repair the body. |
| Non-`PDFUploaded` event | `ack` | Not work for this subscription. |
| Missing `gcs_file_path` | `ack` | Permanent schema error. |
| GCS download failure | `nack` | May be transient and should retry. |
| Processing completed, including a structured failure result | `ack` | Work produced its final event/output. |

Publishing and GCS-upload failures are logged. Local output is retained when
GCS upload fails.

### Validate Pub/Sub connectivity

When `ENABLE_PUBSUB_LISTENER=true`, the API registers `/test-pubsub`:

```bash
curl -fsS http://localhost:8000/test-pubsub
```

The current diagnostic endpoint targets the legacy
`projects/jlr-dl-iqm/topics/iqm_rms_ai` topic. For other projects, validate the
configured resources directly:

```bash
gcloud pubsub subscriptions describe iqm_rms_ai_upload_sub \
  --project=YOUR_PROJECT_ID

gcloud pubsub topics publish iqm_rms_ai \
  --project=YOUR_PROJECT_ID \
  --attribute=event_type=PDFUploaded \
  --message='{
    "event_id":"manual-test",
    "event_type":"PDFUploaded",
    "timestamp":"2026-07-23T10:00:00Z",
    "source":"RMS",
    "payload":{
      "regulation_change_id":12345,
      "document_version":1,
      "gcs_file_path":"gs://YOUR_BUCKET/path/test.pdf"
    }
  }'
```

Follow the listener while testing:

```bash
docker compose logs -f api
```

## Persistent directories

Compose mounts these host directories:

```text
./outputs      final JSON responses
./logs         per-document processing logs
./checkpoints  recoverable intermediate pipeline state
```

They are intentionally excluded from the image and Git.

## Configuration

### Host and container configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_PORT` | `8000` | Host port mapped to FastAPI port 8000. |
| `PADDLEX_MODEL_ROOT` | `./models` | Host model-cache directory mounted into both services. Use an explicit path if the model is stored elsewhere. |
| `UBUNTU_MIRROR` | `http://archive.ubuntu.com/ubuntu` | Ubuntu package mirror used while building the API image. |
| `CLIENT_NAME` | `JLR` | Prefix in generated output filenames. |
| `DEFAULT_MODEL_TYPE` | `vl` | Default OCR path: `vl` or `v3`. |
| `OCR_LANGUAGE` | `en` | Paddle OCR language configuration. |
| `GPU_MEMORY_FRACTION` | `0.7` | Paddle-side memory fraction setting. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.75` | Fraction of GPU memory reserved by vLLM. |
| `VLLM_LOGGING_LEVEL` | `INFO` | vLLM log level; temporarily use `DEBUG` for support diagnostics. |
| `VLLM_MODEL_NAME` | `PaddleOCR-VL-1.5-0.9B` | Served model name requested by the API worker. |

### AI provider configuration

| Variable | Default | Required when |
|---|---|---|
| `AI_PROVIDER` | `openai` | Selects `openai`, `gemini`, or `vertex`. |
| `OPENAI_API_KEY` | empty | OpenAI-backed options are enabled. |
| `GEMINI_API_KEY` | empty | Gemini Developer API options are enabled. |
| `GCP_PROJECT_ID` | empty | Vertex or Google Cloud services are used. |
| `GCP_LOCATION` | `europe-west2` | Vertex AI is used. |
| `VERTEX_MODEL` | `gemini-2.5-flash` | Default Vertex/Gemini analysis model. |

Provider choice can be overridden per HTTP request. Changing providers does
not require restarting the OCR models, but the selected provider must have
valid credentials whenever an AI option is enabled.

### Pub/Sub configuration

| Variable | Default | Purpose |
|---|---|---|
| `ENABLE_PUBSUB_LISTENER` | `false` in `.env.example` | Starts the background pull listener and registers `/test-pubsub`. |
| `PUBSUB_SUBSCRIPTION` | `projects/jlr-dl-iqm/subscriptions/iqm_rms_ai_upload_sub` | Full pull-subscription resource name. |
| `PUBSUB_TOPIC` | `projects/jlr-dl-iqm/topics/iqm_rms_ai` | Full resource name for outgoing events. |
| `PUBSUB_NUM_PAGES` | `0` | `0` processes all pages; a positive value caps automated jobs. |

The application-level fallback for `ENABLE_PUBSUB_LISTENER` is `true`, while
the supplied `.env.example` deliberately sets it to `false` for safe HTTP-only
startup. Copy `.env.example` rather than starting an unconfigured cloud
listener.

### Limits and persistence

| Variable | Default | Purpose |
|---|---:|---|
| `REQUEST_TIMEOUT_S` | `60` | HTTP PDF download timeout. |
| `MAX_DOWNLOAD_MB` | `100` | Maximum HTTP-downloaded PDF size. |
| `CHECKPOINT_MAX_MB` | `500` | Checkpoint storage budget. |
| `LOG_MAX_MB` | `200` | Per-document log storage budget. |

Increase vLLM GPU utilization only after confirming enough memory remains for
the Paddle worker. On a 24 GB GPU, the default 0.75 is safer than reserving 0.85
for vLLM.

## Stop and inspect

```bash
docker compose logs -f api vllm
docker compose down
```

Do not expose vLLM port 8080 publicly. Only the API service needs an external
port.

### Operational inspection

Use these commands before changing configuration:

```bash
# Container state, health and published ports
docker compose ps

# Recent logs from both services
docker compose logs --tail=200 api vllm

# Follow only the API or model server
docker compose logs -f api
docker compose logs -f vllm

# GPU processes and memory use
nvidia-smi

# Docker and host disk use
docker system df
df -h

# Confirm the rendered environment, mounts and port mapping
docker compose config

# Inspect generated data
find outputs logs checkpoints -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TT %s %p\n' \
  | sort
```

Typical healthy startup order:

1. `vllm` starts, loads the model, compiles/captures graphs, and becomes
   `healthy`.
2. Compose starts `api` because its dependency condition is satisfied.
3. FastAPI reports application startup complete.
4. `/health` and `/ready` both return HTTP 200.
5. In Pub/Sub mode, API logs report that the listener thread and subscription
   started.

## Known deployment issue: vLLM restart loop

Use this standalone section when `docker ps -a` or `docker compose ps` shows
the vLLM container repeatedly changing to a state such as:

```text
docker-paddle-ocr-vllm-1   Restarting (1) 8 seconds ago
```

`Restarting (1)` means the vLLM server process exited with status 1. Docker is
relaunching it because the service uses `restart: unless-stopped`. A failed
health check alone does not restart the container.

The API container may still display `Up`. Compose dependency conditions control
startup ordering; they do not automatically stop the API if vLLM crashes
later. `/health` may therefore return 200 while `/ready` returns 503.

### First command: collect the actual vLLM error

Do not rebuild the images first. Capture the process error:

```bash
docker compose logs --no-color --timestamps --tail=200 vllm
```

The last exception normally identifies one of these failure classes:

| Log text or symptom | Likely cause |
|---|---|
| Model path does not exist, invalid repository, or missing `config.json` | Wrong/empty model bind mount. |
| `CUDA driver version is insufficient` | Host NVIDIA driver is incompatible with the container CUDA runtime. |
| No NVIDIA driver, no CUDA device, or failed to infer device | NVIDIA Container Toolkit/runtime is not working. |
| CUDA out of memory or engine-core initialization failure | Insufficient free VRAM or another GPU process is running. |
| Exit code `137` or host OOM messages | Host RAM exhaustion or an external kill. |

### Most common cause: running Compose as root changes `${HOME}`

Older copies of `.env.example` used:

```text
PADDLEX_MODEL_ROOT=${HOME}/.paddlex/official_models
```

If a customer downloads the model as a normal user but later runs
`sudo docker compose` or works in a root shell, `${HOME}` becomes `/root`.
Compose then mounts `/root/.paddlex/official_models` instead of the actual
directory under `/home/<user>`. Docker may create an empty host directory, so
the container starts without `/models/PaddleOCR-VL-1.5` and immediately exits.

Show the effective mount:

```bash
docker inspect "$(docker compose ps -q vllm)" \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

grep '^PADDLEX_MODEL_ROOT=' .env
```

The current repository avoids user-dependent expansion by using:

```text
PADDLEX_MODEL_ROOT=./models
```

For an existing installation, either place the model in `./models` or set the
exact absolute host path. For example:

```text
PADDLEX_MODEL_ROOT=/home/customer/.paddlex/official_models
```

Verify that the configured root contains the required subdirectory:

```bash
ls -lh ./models/PaddleOCR-VL-1.5/config.json
ls -lh ./models/PaddleOCR-VL-1.5/model.safetensors
```

When using an absolute path, replace `./models` in those checks with that path.

Apply the corrected mount without rebuilding:

```bash
docker compose up -d --force-recreate vllm
docker compose logs -f vllm
```

After vLLM reports healthy:

```bash
docker compose up -d api
docker compose ps
curl -i http://localhost:8000/ready
```

### Verify the NVIDIA runtime

If the model mount is correct, test the host and the exact vLLM image:

```bash
nvidia-smi

docker run --rm --gpus all \
  nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  docker-paddle-ocr-vllm:latest
```

All three commands must display the intended GPU. If the CUDA container fails,
fix the host driver/NVIDIA Container Toolkit before changing application code.

### Check GPU memory

The tested profile is a 24 GB NVIDIA A10G. Check for other GPU consumers and
vLLM memory errors:

```bash
nvidia-smi
docker compose logs --no-color vllm \
  | grep -iE 'out of memory|oom|engine core|cuda'
```

If Compose is the only GPU workload but vLLM cannot allocate memory, try:

```text
VLLM_GPU_MEMORY_UTILIZATION=0.70
```

Then recreate vLLM. Lowering this value leaves more VRAM for the Paddle worker
but reduces vLLM KV-cache capacity.

### Enable vLLM debug logging

Set this temporarily in `.env`:

```text
VLLM_LOGGING_LEVEL=DEBUG
```

Recreate and reproduce the failure:

```bash
docker compose up -d --force-recreate vllm
docker compose logs --no-color --timestamps -f vllm
```

`VLLM_LOGGING_LEVEL` is supported by the pinned vLLM 0.19 runtime. Return it to
`INFO` after diagnosis because debug logs are substantially noisier.

### Final customer log request

If the issue remains, ask the customer to run this block from the repository
directory and send back `vllm-support.log`:

```bash
{
  echo '=== timestamp ==='
  date -u

  echo '=== Docker versions ==='
  docker version
  docker compose version

  echo '=== Compose state ==='
  docker compose ps -a

  VLLM_ID="$(docker compose ps -aq vllm)"
  echo "VLLM_ID=${VLLM_ID}"

  echo '=== Exit state ==='
  docker inspect "${VLLM_ID}" \
    --format 'State={{json .State}} Image={{.Config.Image}}'

  echo '=== Effective mounts ==='
  docker inspect "${VLLM_ID}" \
    --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

  echo '=== Safe vLLM configuration ==='
  grep -E '^(PADDLEX_MODEL_ROOT|VLLM_MODEL_NAME|VLLM_GPU_MEMORY_UTILIZATION|VLLM_LOGGING_LEVEL)=' \
    .env || true

  MODEL_SOURCE="$(docker inspect "${VLLM_ID}" \
    --format '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Source}}{{end}}{{end}}')"
  echo "MODEL_SOURCE=${MODEL_SOURCE}"

  echo '=== Model directory ==='
  ls -lah "${MODEL_SOURCE}/PaddleOCR-VL-1.5" || true

  echo '=== Host GPU ==='
  nvidia-smi || true

  echo '=== GPU from the vLLM image ==='
  VLLM_IMAGE="$(docker inspect "${VLLM_ID}" --format '{{.Config.Image}}')"
  docker run --rm --gpus all --entrypoint nvidia-smi "${VLLM_IMAGE}" || true

  echo '=== Last 500 vLLM log lines ==='
  docker compose logs --no-color --timestamps --tail=500 vllm
} >vllm-support.log 2>&1
```

This bundle deliberately excludes the full Compose environment so API keys and
cloud credentials are not collected. Customers should still review the file
and redact private hostnames, usernames, bucket names, or document paths before
sharing it.

## Troubleshooting

### Establish the current failure boundary

Run:

```bash
docker compose ps
curl -i http://localhost:8000/health
curl -i http://localhost:8000/ready
docker compose logs --tail=200 vllm
docker compose logs --tail=200 api
```

Interpretation:

| Result | Likely boundary |
|---|---|
| `/health` is unreachable | API container, port mapping, or startup failure. |
| `/health` is 200 and `/ready` is 503 | vLLM is unavailable or not healthy. |
| Both are 200 but OCR fails | PDF input, Paddle worker, provider, or pipeline. |
| HTTP works but automated jobs do not | Pub/Sub listener, IAM, subscription, or GCS. |
| Output exists locally but not in GCS | Storage credentials/permission or upload failure. |

### Docker cannot select the NVIDIA driver

Re-run the toolkit configuration, restart Docker, and repeat the CUDA-container
test from step 1:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Also verify:

```bash
nvidia-container-cli --version
docker info | grep -i runtime
```

### API port 8000 is already in use

Identify the listener and choose a different host port rather than changing the
container port:

```bash
ss -ltnp | grep ':8000'
```

Set in `.env`:

```text
API_PORT=8001
```

Then recreate the API service and use `http://localhost:8001`:

```bash
docker compose up -d api
```

### API is waiting for vLLM or `/ready` returns 503

The API will not start until the Compose vLLM health check succeeds. Model
loading and first-time graph compilation can take several minutes:

```bash
docker compose ps
docker compose logs -f vllm
docker inspect "$(docker compose ps -q vllm)" \
  --format '{{json .State.Health}}'
```

Look for model-path errors, CUDA errors, an out-of-memory failure, or the vLLM
HTTP-server startup message.

### vLLM reports that the model path does not exist

Check the host path configured in `.env`:

```bash
grep '^PADDLEX_MODEL_ROOT=' .env
ls ./models/PaddleOCR-VL-1.5
docker compose config | grep -A2 '/models'
```

The directory mounted at `/models` must contain a `PaddleOCR-VL-1.5`
subdirectory.

### Served model name mismatch

The vLLM `--served-model-name` and the name requested by PaddleOCR must match:

```bash
docker compose exec -T vllm \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/v1/models').read().decode())"

grep '^VLLM_MODEL_NAME=' .env
```

The tested value is:

```text
PaddleOCR-VL-1.5-0.9B
```

### GPU out of memory

Confirm which processes own GPU memory:

```bash
nvidia-smi
docker compose logs --tail=200 api vllm | grep -iE 'out of memory|oom|cuda'
```

Stop any unrelated GPU process. If the Compose stack is the only consumer,
lower `VLLM_GPU_MEMORY_UTILIZATION` in `.env`, recreate vLLM, and retest:

```text
VLLM_GPU_MEMORY_UTILIZATION=0.70
```

```bash
docker compose up -d --force-recreate vllm
docker compose up -d api
```

Lower values leave more memory for the Paddle layout worker but reduce vLLM KV
cache capacity. The tested 24 GB profile uses `0.75`.

### Paddle model download is interrupted

The API downloads layout-model assets into `PADDLEX_MODEL_ROOT` on demand. If a
first request is interrupted, inspect for incomplete temporary directories:

```bash
find ./models -type d -name temp_dir -print
docker compose logs api | grep -iE 'download|temp_dir|model source'
```

Stop the API before removing only a confirmed incomplete `temp_dir`, then
restart with outbound network access. Do not remove the
`PaddleOCR-VL-1.5` directory used by vLLM.

### PDF download fails

The HTTP endpoint requires a directly accessible HTTP(S) PDF URL:

```bash
curl -I -L https://example.com/document.pdf
```

Check redirects, authentication, response size, `REQUEST_TIMEOUT_S`, and
`MAX_DOWNLOAD_MB`. Increase the limits in `.env` only when the source is
trusted:

```text
REQUEST_TIMEOUT_S=120
MAX_DOWNLOAD_MB=250
```

Recreate the API after changing container environment:

```bash
docker compose up -d --force-recreate api
```

For `gs://` inputs, use Pub/Sub mode; `/doc-intel` does not directly download a
`gs://` URL.

### Pipeline failure is returned with HTTP 200

Processing exceptions are converted to the standard event schema. Inspect:

```bash
jq '.payload.processing_status, .payload.error_message' response.json
docker compose logs --tail=300 api
```

Treat `payload.processing_status` as the processing outcome rather than relying
only on the HTTP status code.

### OpenAI or Gemini authentication fails

Confirm that the relevant value is present inside the API container without
printing the secret:

```bash
docker compose exec -T api python3 -c \
  'import os; print(bool(os.getenv("OPENAI_API_KEY")), bool(os.getenv("GEMINI_API_KEY")))'
```

Only enabled AI table, topic, or summary operations require the provider
credential. Recreate the API after changing `.env`.

### Vertex AI fails with 403, 404, or 429

- `403`: verify ADC and `roles/aiplatform.user`.
- `404`: verify `GCP_LOCATION` and that `VERTEX_MODEL` is available there.
- `429`: inspect Vertex AI quotas for the project and region.

On GCE, inspect the attached service-account identity:

```bash
curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email

gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten='bindings[].members' \
  --filter='bindings.role:roles/aiplatform.user'
```

### Pub/Sub listener does not start

```bash
grep '^ENABLE_PUBSUB_LISTENER=' .env
docker compose config | grep -A3 ENABLE_PUBSUB_LISTENER
docker compose logs api | grep -iE 'pub.?sub|subscription|listener'
```

The API must be recreated after enabling the listener:

```bash
docker compose up -d --force-recreate api
```

### Pub/Sub returns permission denied

Verify the actual container identity and the two Pub/Sub roles:

```bash
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten='bindings[].members' \
  --filter='bindings.role:roles/pubsub.subscriber OR bindings.role:roles/pubsub.publisher'
```

Also check permissions on the specific topic and subscription when IAM is
resource-scoped.

### Pub/Sub subscription is not found

```bash
gcloud pubsub subscriptions describe iqm_rms_ai_upload_sub \
  --project=YOUR_PROJECT_ID

gcloud pubsub subscriptions list \
  --project=YOUR_PROJECT_ID
```

Ensure `.env` contains the full resource path, including the correct project.

### GCS download or upload fails

Confirm the object and test access using the same service-account context:

```bash
gcloud storage ls gs://YOUR_BUCKET/path/document.pdf
docker compose logs api | grep -iE 'GCS|storage|permission|403'
```

The service needs object-viewer permission for input and object-creator
permission for output. If upload fails, the full JSON remains in `./outputs`.
The current writer logs the failure but leaves `processed_bucket_path` as the
intended GCS URI, so operators must resolve upload errors before downstream
consumers fetch that path.

The publisher helper also logs publish failures without crashing the processing
job. Alert on `Failed to publish event` so a locally completed job is not
mistaken for successful event delivery.

### Build fails with no space left on device

Check disk usage with `df -h` and `docker system df`. Preserve at least 70 GB
before the initial build. Download caches such as `pip cache` can be removed
with `python3 -m pip cache purge`; do not delete model or output directories.

### Ubuntu package downloads time out during the API build

Set `UBUNTU_MIRROR` in `.env` to a responsive Ubuntu mirror for the deployment
region and rebuild the API image:

```bash
docker compose build api
docker compose up -d
```

### Output, log, or checkpoint growth

Inspect the mounted host directories:

```bash
du -sh outputs logs checkpoints
find outputs logs checkpoints -type f -printf '%s %p\n' | sort -n | tail
```

The application applies configured checkpoint/log budgets, but final outputs
remain until an operator archives or removes them. Back up required results
before any cleanup.

## Performance and operating notes

- Start validation with one page and all optional AI features disabled.
- Increase `num_pages` only after confirming stable GPU and disk usage.
- `enable_ai_tables=false` skips provider calls for table enrichment.
- `enable_topic_ai=false` keeps deterministic rule-derived topic labels.
- `do_summary=false` avoids the final AI briefing call.
- Jobs are intentionally serialized; increasing HTTP or Pub/Sub concurrency
  does not increase GPU pipeline concurrency.
- `PUBSUB_NUM_PAGES=0` means the full PDF and can create long-running jobs.
- vLLM graph compilation makes the first startup slower than later starts.
- Layout models are cached in `PADDLEX_MODEL_ROOT`; do not use a read-only mount
  for the API side of that cache.
- `./outputs`, `./logs`, and `./checkpoints` are host-mounted and survive
  `docker compose down`.
- API responses contain the full extraction. Pub/Sub completion events contain
  metadata only to keep message size bounded.

## License

Internal JLR development build.
