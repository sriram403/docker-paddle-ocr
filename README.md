# JLR Document Intelligence API

GPU-accelerated regulatory-document extraction based on the current
`experimentation` pipeline. The production interface remains FastAPI with
optional Google Cloud Pub/Sub input and GCS output.

The service extracts document hierarchy, clauses, tables, figures, references,
topic labels, requirement labels, text-type labels, and an optional regulatory
briefing.

## Runtime architecture

The deployment uses two GPU-enabled services:

```text
HTTP / Pub/Sub -> FastAPI + Paddle layout worker -> PaddleOCR-VL vLLM server
                       |                                  |
                       +---- outputs/logs/checkpoints ----+
```

- `api` exposes port 8000, listens for Pub/Sub events, and runs the isolated
  Paddle layout worker.
- `vllm` serves `PaddleOCR-VL-1.5-0.9B` internally on port 8080.
- Jobs are serialized inside the API process to protect shared GPU memory.
- Pub/Sub flow control also limits the subscriber to one outstanding message.

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

export PADDLEX_MODEL_ROOT="$HOME/.paddlex/official_models"
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

Review `.env` and set `PADDLEX_MODEL_ROOT` to the absolute model-cache path.
Leave `ENABLE_PUBSUB_LISTENER=false` for HTTP-only deployments. Add credentials
only for the AI provider and optional features being used.

For example, an OpenAI HTTP-only deployment needs:

```bash
AI_PROVIDER=openai
OPENAI_API_KEY=your-key
ENABLE_PUBSUB_LISTENER=false
PADDLEX_MODEL_ROOT=/home/your-user/.paddlex/official_models
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

## Pub/Sub mode

Set `ENABLE_PUBSUB_LISTENER=true` and configure:

```text
PUBSUB_SUBSCRIPTION
PUBSUB_TOPIC
PUBSUB_NUM_PAGES
GCP_PROJECT_ID
GCP_LOCATION
```

The listener accepts `PDFUploaded`, publishes `FileProcessingStarted`, downloads
the source PDF from GCS, writes the full result locally and to GCS, and publishes
a metadata-only `FileProcessingCompleted` event. Existing event and output
naming contracts are retained.

`PUBSUB_NUM_PAGES=0` processes the full PDF. Set a positive value to cap
automated jobs at that many pages.

## Persistent directories

Compose mounts these host directories:

```text
./outputs      final JSON responses
./logs         per-document processing logs
./checkpoints  recoverable intermediate pipeline state
```

They are intentionally excluded from the image and Git.

## Configuration

| Variable | Default |
|---|---|
| `API_PORT` | `8000` |
| `DEFAULT_MODEL_TYPE` | `vl` |
| `OCR_LANGUAGE` | `en` |
| `GPU_MEMORY_FRACTION` | `0.7` |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.75` |
| `VLLM_MODEL_NAME` | `PaddleOCR-VL-1.5-0.9B` |
| `VERTEX_MODEL` | `gemini-2.5-flash` |
| `CHECKPOINT_MAX_MB` | `500` |
| `LOG_MAX_MB` | `200` |
| `MAX_DOWNLOAD_MB` | `100` |
| `REQUEST_TIMEOUT_S` | `60` |

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

## Troubleshooting

### Docker cannot select the NVIDIA driver

Re-run the toolkit configuration, restart Docker, and repeat the CUDA-container
test from step 1:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### vLLM reports that the model path does not exist

Check the host path configured in `.env`:

```bash
grep '^PADDLEX_MODEL_ROOT=' .env
ls "$HOME/.paddlex/official_models/PaddleOCR-VL-1.5"
docker compose config | grep -A2 '/models'
```

The directory mounted at `/models` must contain a `PaddleOCR-VL-1.5`
subdirectory.

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
