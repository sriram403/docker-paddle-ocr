import os

CLIENT_NAME = os.getenv("CLIENT_NAME", "JLR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/outputs")
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "60"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "100"))
