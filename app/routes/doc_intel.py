import httpx
from fastapi import APIRouter, Body
from app.core.config import REQUEST_TIMEOUT_S, MAX_DOWNLOAD_MB, AI_PROVIDER
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

    pdf_bytes = download_pdf(document_url)

    # Get AI provider from payload, fallback to env var
    ai_provider = payload.get("ai_provider", AI_PROVIDER).lower().strip()

    # Validate provider
    if ai_provider not in ["openai", "vertex"]:
        return {"status": "failed", "error": f"Invalid ai_provider: {ai_provider}. Use 'openai' or 'vertex'"}

    logger.info(f"Processing document with AI provider: {ai_provider}")

    response = service.process_document(
        pdf_bytes=pdf_bytes,
        num_pages=payload.get("num_pages", 10),
        enable_ai_tables=payload.get("enable_ai_tables", False),
        analysis_model=payload.get("analysis_model", "gpt-4o-mini"),
        do_summary=payload.get("do_summary", False),
        model_type=payload.get("model_type", "v3"),
        ai_provider=ai_provider,
        regulation_change_id=payload.get("regulation_change_id"),
        document_version=payload.get("document_version"),
        gcs_file_path=payload.get("gcs_file_path") or document_url
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
