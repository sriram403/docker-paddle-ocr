#!/usr/bin/env python3
"""Run a small end-to-end extraction without starting FastAPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.engine.doc_processor_engine import DocumentProcessor, shutdown_paddle_worker
from app.services.doc_intel_service import DocIntelService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--model-type", choices=("vl", "v3"), default="vl")
    args = parser.parse_args()

    service = DocIntelService(DocumentProcessor)
    try:
        response = service.process_document(
            pdf_bytes=args.pdf.read_bytes(),
            num_pages=args.pages,
            enable_ai_tables=False,
            analysis_model="gpt-4o-mini",
            do_summary=False,
            model_type=args.model_type,
            document_name=args.pdf.name,
            enable_topic_ai=False,
            enable_checkpoint=False,
        )
    finally:
        shutdown_paddle_worker()

    payload = response["payload"]
    # The API writer and Pub/Sub publisher both require the entire response to
    # be JSON serializable, not only its metadata.
    json.dumps(response)
    print(json.dumps(payload["summary_metadata"], indent=2))
    if payload["processing_status"] != "SUCCESS":
        print(payload["error_message"])
        return 1
    if not payload["data"]["results"]:
        print("Extraction returned no chunks")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
