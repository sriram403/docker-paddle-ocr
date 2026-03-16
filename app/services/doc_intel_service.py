import fitz
import time
import uuid
from datetime import datetime
from app.core.config import OPENAI_API_KEY
from app.core.logging import get_logger
from app.core.model_registry import get_paddle_model

logger = get_logger("doc_intel_service")


class DocIntelService:

    def __init__(self, processor_class, paddle_loader):
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
        regulation_change_id: int = None,
        document_version: int = None,
        gcs_file_path: str = None
    ):
        start_time = time.time()

        try:
            processor = self.processor_class(ai_provider=ai_provider)
            processor.openai_api_key = OPENAI_API_KEY  # Keep for backward compatibility

            paddle_model = get_paddle_model()

            result, metrics, _, summary = processor.run_pipeline(
                pdf_bytes,
                num_pages,
                enable_ai_tables,
                110,
                table_model="gpt-4o",
                analysis_model=analysis_model,
                api_key=OPENAI_API_KEY,
                do_summary=do_summary,
                paddle_model=paddle_model,
                model_type=model_type
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            # 🔥 remove non-serializable debug image bytes
            if metrics and "debug_images" in metrics:
                metrics["debug_images"] = {}

            # Count total chunks and labels
            total_chunks = len(result) if result else 0
            unique_labels = set()
            if result:
                for chunk in result:
                    if chunk.get("Requirement Label"):
                        unique_labels.add(chunk["Requirement Label"])

            # Build Pub/Sub schema compliant response (FileProcessingCompleted event)
            response = {
                "event_id": str(uuid.uuid4()),
                "event_type": "FileProcessingCompleted",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "RMS AI",
                "payload": {
                    "regulation_change_id": regulation_change_id,
                    "document_version": document_version,
                    "processed_bucket_path": gcs_file_path,  # Will be updated by output_writer
                    "processing_status": "SUCCESS",
                    "processing_time_ms": processing_time_ms,
                    "error_message": None,
                    "summary_metadata": {
                        "pages_processed": num_pages,
                        "confidence_score": metrics.get("kept_words", 0) / max(metrics.get("total_words", 1), 1) if metrics else 0,
                        "total_chunks_processed": total_chunks,
                        "unique_labels_detected": len(unique_labels),
                        "ai_provider_used": ai_provider,
                        "model_type": model_type,
                        "analysis_model": analysis_model,
                        "tables_detected": len(metrics.get("detected_tables", {})) if metrics else 0,
                        "summary_generated": bool(summary)
                    },
                    # Full structured data (backward compatibility)
                    "data": {
                        "results": result,
                        "metrics": metrics,
                        "summary": summary
                    }
                }
            }

            return response

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(str(e))

            # Build Pub/Sub schema compliant error response
            return {
                "event_id": str(uuid.uuid4()),
                "event_type": "FileProcessingCompleted",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "RMS AI",
                "payload": {
                    "regulation_change_id": regulation_change_id,
                    "document_version": document_version,
                    "processed_bucket_path": gcs_file_path,
                    "processing_status": "FAILURE",
                    "processing_time_ms": processing_time_ms,
                    "error_message": str(e),
                    "summary_metadata": {
                        "pages_processed": 0,
                        "confidence_score": 0.0,
                        "total_chunks_processed": 0,
                        "unique_labels_detected": 0,
                        "ai_provider_used": ai_provider,
                        "model_type": model_type
                    }
                }
            }
