import os

CLIENT_NAME = os.getenv("CLIENT_NAME", "JLR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/outputs")
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr")

# AI Provider Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "openai" or "vertex"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Vertex AI Configuration (GCP)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west2")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-exp")

REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "60"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "100"))
