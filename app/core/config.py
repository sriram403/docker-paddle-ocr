import os

CLIENT_NAME = os.getenv("CLIENT_NAME", "JLR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/outputs")
LOG_DIR = os.getenv("LOG_DIR", "/logs")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/checkpoints")
CHECKPOINT_MAX_MB = int(os.getenv("CHECKPOINT_MAX_MB", "500"))
LOG_MAX_MB = int(os.getenv("LOG_MAX_MB", "200"))
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr")
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "vl")
GPU_MEMORY_FRACTION = os.getenv("GPU_MEMORY_FRACTION", "0.7")
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://vllm:8080/v1")
VLLM_HEALTH_URL = os.getenv("VLLM_HEALTH_URL", "http://vllm:8080/health")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "PaddleOCR-VL-1.5-0.9B")

# AI Provider Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "openai" or "vertex"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Vertex AI Configuration (GCP)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west2")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")

REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "60"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "100"))

# Pub/Sub Configuration
PUBSUB_SUBSCRIPTION = os.getenv("PUBSUB_SUBSCRIPTION", "projects/jlr-dl-iqm/subscriptions/iqm_rms_ai_upload_sub")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "projects/jlr-dl-iqm/topics/iqm_rms_ai")
PUBSUB_NUM_PAGES = int(os.getenv("PUBSUB_NUM_PAGES", "0"))
ENABLE_PUBSUB_LISTENER = os.getenv("ENABLE_PUBSUB_LISTENER", "true").lower() == "true"
