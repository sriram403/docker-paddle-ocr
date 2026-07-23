import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import (
    CHECKPOINT_DIR,
    GEMINI_API_KEY,
    GCP_PROJECT_ID,
    OPENAI_API_KEY,
)
from app.core.logging import get_logger


logger = get_logger("doc_intel_service")

# A single GPU worker/model server is shared by HTTP and Pub/Sub entry points.
# Serializing jobs prevents concurrent requests from exhausting GPU memory or
# corrupting the pipeline's reusable worker state.
_PROCESSING_LOCK = threading.Lock()


def _api_key_for(provider: str) -> str:
    if provider == "openai":
        return OPENAI_API_KEY
    if provider == "gemini":
        return GEMINI_API_KEY
    return ""  # Vertex uses Application Default Credentials.


def _checkpoint_path(pdf_bytes: bytes, num_pages: int, page_start: int) -> str:
    digest = hashlib.sha256(pdf_bytes[:8192]).hexdigest()[:16]
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(CHECKPOINT_DIR) / f"ckpt_{digest}_{page_start}_{num_pages}.json")


def _resume_page(path: str | None, page_start: int) -> int:
    if not path or not Path(path).is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
        # The pipeline stores p_idx + 1, which is both the human page number
        # just completed and the zero-based index of the next page.
        return max(page_start, int(checkpoint["last_completed_page"]))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable checkpoint: %s", path)
        return 0


class DocIntelService:
    def __init__(self, processor_class, paddle_loader=None):
        self.processor_class = processor_class
        self.paddle_loader = paddle_loader

    def process_document(
        self,
        pdf_bytes: bytes,
        num_pages: int,
        enable_ai_tables: bool,
        analysis_model: str,
        do_summary: bool,
        model_type: str,
        ai_provider: str = "openai",
        regulation_change_id: int | None = None,
        document_version: int | None = None,
        gcs_file_path: str | None = None,
        document_name: str = "unknown.pdf",
        page_start: int = 0,
        enable_topic_ai: bool = False,
        enable_checkpoint: bool = True,
        table_model: str | None = None,
    ):
        started = time.monotonic()
        api_key = _api_key_for(ai_provider)
        checkpoint_path = (
            _checkpoint_path(pdf_bytes, num_pages, page_start)
            if enable_checkpoint
            else None
        )
        resume_from_page = _resume_page(checkpoint_path, page_start)

        try:
            ai_requested = enable_topic_ai or enable_ai_tables or do_summary
            if ai_provider == "openai" and ai_requested and not api_key:
                raise ValueError("OPENAI_API_KEY is required for enabled OpenAI operations")
            if ai_provider == "gemini" and ai_requested and not api_key:
                raise ValueError("GEMINI_API_KEY is required for enabled Gemini operations")
            if ai_provider == "vertex" and ai_requested and not GCP_PROJECT_ID:
                raise ValueError("GCP_PROJECT_ID is required for enabled Vertex operations")

            processor = self.processor_class(
                ai_provider=ai_provider,
                document_name=document_name,
            )

            with _PROCESSING_LOCK:
                result, metrics, _, summary = processor.run_pipeline(
                    pdf_bytes=pdf_bytes,
                    num_pages=num_pages,
                    enable_ai=enable_ai_tables,
                    ai_thresh=110,
                    table_model=table_model or (
                        "gpt-4o" if ai_provider == "openai" else "gemini-2.5-flash"
                    ),
                    analysis_model=analysis_model,
                    api_key=api_key,
                    paddle_model=None,
                    model_type=model_type,
                    ai_provider=ai_provider,
                    checkpoint_path=checkpoint_path,
                    resume_from_page=resume_from_page,
                    page_start=page_start,
                    enable_topic_ai=enable_topic_ai,
                    enable_ai_summary=do_summary,
                )

            metrics = metrics or {}
            table_count = sum(
                len(rects) for rects in metrics.get("detected_tables", {}).values()
            )
            metrics.pop("debug_images", None)
            metrics.pop("detected_tables", None)
            metrics.pop("detected_columnar_content", None)

            unique_labels = {
                label
                for chunk in result or []
                for label in (
                    chunk.get("requirement_label_1"),
                    chunk.get("requirement_label_2"),
                )
                if label and label != "Unclassified"
            }
            elapsed_ms = int((time.monotonic() - started) * 1000)

            return {
                "event_id": str(uuid.uuid4()),
                "event_type": "FileProcessingCompleted",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": "RMS AI",
                "payload": {
                    "regulation_change_id": regulation_change_id,
                    "document_version": document_version,
                    "processed_bucket_path": gcs_file_path,
                    "processing_status": "SUCCESS",
                    "processing_time_ms": elapsed_ms,
                    "error_message": None,
                    "summary_metadata": {
                        "pages_processed": max(0, num_pages - page_start),
                        "confidence_score": metrics.get("kept_words", 0)
                        / max(metrics.get("total_words", 1), 1),
                        "total_chunks_processed": len(result or []),
                        "unique_labels_detected": len(unique_labels),
                        "ai_provider_used": ai_provider,
                        "model_type": model_type,
                        "analysis_model": analysis_model,
                        "tables_detected": table_count,
                        "summary_generated": bool(summary),
                    },
                    "data": {
                        "results": result or [],
                        "metrics": metrics,
                        "summary": summary or {},
                    },
                },
            }
        except Exception as error:
            logger.exception("Document processing failed")
            return {
                "event_id": str(uuid.uuid4()),
                "event_type": "FileProcessingCompleted",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": "RMS AI",
                "payload": {
                    "regulation_change_id": regulation_change_id,
                    "document_version": document_version,
                    "processed_bucket_path": gcs_file_path,
                    "processing_status": "FAILURE",
                    "processing_time_ms": int((time.monotonic() - started) * 1000),
                    "error_message": str(error),
                    "summary_metadata": {
                        "pages_processed": 0,
                        "confidence_score": 0.0,
                        "total_chunks_processed": 0,
                        "unique_labels_detected": 0,
                        "ai_provider_used": ai_provider,
                        "model_type": model_type,
                    },
                },
            }
