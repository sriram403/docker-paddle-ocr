import httpx
from fastapi import APIRouter, Body
from urllib.parse import urlparse

from app.core.config import (
    AI_PROVIDER,
    DEFAULT_MODEL_TYPE,
    MAX_DOWNLOAD_MB,
    REQUEST_TIMEOUT_S,
    VERTEX_MODEL,
)
from app.core.logging import get_logger
from app.services.doc_intel_service import DocIntelService
from app.storage.output_writer import save_response

from app.engine.doc_processor_engine import DocumentProcessor, load_paddle_model


router = APIRouter()
logger = get_logger("doc_intel_route")

service = DocIntelService(DocumentProcessor, load_paddle_model)


def download_pdf(url: str) -> bytes:
    with httpx.Client(timeout=REQUEST_TIMEOUT_S, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()

        content = r.content
        size_mb = len(content) / (1024 * 1024)

        if size_mb > MAX_DOWNLOAD_MB:
            raise RuntimeError("File too large")

        return content


@router.post("/doc-intel")
def doc_intel_endpoint(payload: dict = Body(...)):

    document_url = payload.get("document_url")
    request_id = payload.get("request_id")

    if not document_url:
        return {"status": "failed", "error": "document_url missing"}

    try:
        num_pages = int(payload.get("num_pages", 10))
        page_start = int(payload.get("page_start", 0))
    except (TypeError, ValueError):
        return {"status": "failed", "error": "num_pages and page_start must be integers"}
    if num_pages < 1 or page_start < 0 or page_start >= num_pages:
        return {
            "status": "failed",
            "error": "Require num_pages >= 1 and 0 <= page_start < num_pages",
        }

    model_type = str(payload.get("model_type", DEFAULT_MODEL_TYPE)).lower().strip()
    if model_type not in ("vl", "v3"):
        return {"status": "failed", "error": "model_type must be 'vl' or 'v3'"}

    pdf_bytes = download_pdf(document_url)

    # Get AI provider from payload, fallback to env var
    ai_provider = payload.get("ai_provider", AI_PROVIDER).lower().strip()

    # Validate provider
    if ai_provider not in ["openai", "gemini", "vertex"]:
        return {
            "status": "failed",
            "error": (
                f"Invalid ai_provider: {ai_provider}. "
                "Use 'openai', 'gemini', or 'vertex'"
            ),
        }

    logger.info(f"Processing document with AI provider: {ai_provider}")

    response = service.process_document(
        pdf_bytes=pdf_bytes,
        num_pages=num_pages,
        enable_ai_tables=payload.get("enable_ai_tables", False),
        analysis_model=payload.get("analysis_model") or (
            "gpt-4o-mini" if ai_provider == "openai" else VERTEX_MODEL
        ),
        do_summary=payload.get("do_summary", False),
        model_type=model_type,
        ai_provider=ai_provider,
        regulation_change_id=payload.get("regulation_change_id"),
        document_version=payload.get("document_version"),
        gcs_file_path=payload.get("gcs_file_path") or document_url,
        document_name=payload.get("document_name")
        or urlparse(document_url).path.rsplit("/", 1)[-1]
        or "unknown.pdf",
        page_start=page_start,
        enable_topic_ai=payload.get("enable_topic_ai", False),
        enable_checkpoint=payload.get("enable_checkpoint", True),
        table_model=payload.get("table_model"),
    )

    # Add request_id to payload for tracking (not in Pub/Sub schema but useful)
    if request_id:
        response["request_id"] = request_id

    response = save_response(
        feature="DOCINTEL",
        response=response,
        request_id=request_id
    )

    return response
