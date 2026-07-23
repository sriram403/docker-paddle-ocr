import json
import unittest

from app.services.doc_intel_service import DocIntelService


class FakeProcessor:
    last_kwargs = None

    def __init__(self, ai_provider, document_name):
        self.ai_provider = ai_provider
        self.document_name = document_name

    def run_pipeline(self, **kwargs):
        type(self).last_kwargs = kwargs
        chunks = [{
            "chunk_id": 1,
            "requirement_label_1": "Testing",
            "requirement_label_2": "Physical Validation",
        }]
        metrics = {
            "total_words": 10,
            "kept_words": 8,
            "debug_images": {1: b"not serializable"},
            "detected_tables": {1: [object(), object()]},
            "detected_columnar_content": {1: [object()]},
        }
        return chunks, metrics, None, {"briefing": "summary"}


class DocIntelServiceTests(unittest.TestCase):
    def test_current_pipeline_contract_is_wrapped_in_event_schema(self):
        service = DocIntelService(FakeProcessor)
        response = service.process_document(
            pdf_bytes=b"%PDF-test",
            num_pages=3,
            page_start=1,
            enable_ai_tables=False,
            analysis_model="gpt-4o-mini",
            do_summary=False,
            model_type="vl",
            document_name="test.pdf",
            enable_topic_ai=False,
            enable_checkpoint=False,
        )

        payload = response["payload"]
        self.assertEqual("SUCCESS", payload["processing_status"])
        self.assertEqual(2, payload["summary_metadata"]["pages_processed"])
        self.assertEqual(2, payload["summary_metadata"]["tables_detected"])
        self.assertEqual(2, payload["summary_metadata"]["unique_labels_detected"])
        self.assertNotIn("debug_images", payload["data"]["metrics"])
        self.assertNotIn("detected_tables", payload["data"]["metrics"])
        json.dumps(response)

        kwargs = FakeProcessor.last_kwargs
        self.assertEqual("vl", kwargs["model_type"])
        self.assertEqual(1, kwargs["page_start"])
        self.assertFalse(kwargs["enable_topic_ai"])


if __name__ == "__main__":
    unittest.main()
