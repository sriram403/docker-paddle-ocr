import fitz
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
        model_type: str
    ):
        try:
            processor = self.processor_class()
            processor.openai_api_key = OPENAI_API_KEY

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
            # 🔥 remove non-serializable debug image bytes
            if metrics and "debug_images" in metrics:
                metrics["debug_images"] = {}


            response = {
                "request_id": None,
                "created_at": datetime.utcnow().isoformat(),
                "status": "success",
                "data": {
                    "results": result,
                    "metrics": metrics,
                    "summary": summary
                }
            }

            return response

        except Exception as e:
            logger.error(str(e))

            return {
                "request_id": None,
                "created_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": str(e)
            }
