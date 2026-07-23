import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import DEFAULT_MODEL_TYPE, VLLM_HEALTH_URL

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    if DEFAULT_MODEL_TYPE != "vl":
        return {"status": "ready", "vision_backend": DEFAULT_MODEL_TYPE}

    try:
        response = httpx.get(VLLM_HEALTH_URL, timeout=5.0)
        response.raise_for_status()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"PaddleOCR-VL server is not ready: {error}",
        ) from error
    return {"status": "ready", "vision_backend": "vl"}
