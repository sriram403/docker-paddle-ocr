import os
import json
from datetime import datetime
from pathlib import Path
import uuid

from app.core.config import OUTPUT_DIR, CLIENT_NAME


def _timestamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def _generate_job_id():
    return f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def build_filename(feature: str, request_id: str | None):
    ts = _timestamp()
    job_id = _generate_job_id()
    req = request_id if request_id else "NA"

    filename = f"{CLIENT_NAME}__{feature.upper()}__{ts}__{job_id}__{req}.json"
    return filename, job_id


def atomic_write_json(data: dict, filepath: str):
    tmp_path = filepath + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, filepath)


def save_response(feature: str, response: dict, request_id: str | None = None):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    filename, job_id = build_filename(feature, request_id)
    full_path = os.path.join(OUTPUT_DIR, filename)

    # Update Pub/Sub schema fields if present
    if "payload" in response:
        response["payload"]["processed_bucket_path"] = full_path

    # Add legacy fields for backward compatibility
    response["job_id"] = job_id
    response["output_file"] = full_path

    atomic_write_json(response, full_path)

    return response
