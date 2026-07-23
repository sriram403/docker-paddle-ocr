import os
import json
from datetime import datetime
from pathlib import Path
import uuid

from app.core.config import OUTPUT_DIR, CLIENT_NAME
from app.core.logger import get_logger

logger = get_logger("output_writer")


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


def _derive_gcs_output_path(gcs_input_path: str, filename: str) -> tuple[str, str]:
    """Derive GCS output path from the input PDF path.

    gs://bucket/regulations/doc.pdf -> (bucket, regulations/output/filename.json)
    gs://bucket/doc.pdf             -> (bucket, output/filename.json)
    """
    path = gcs_input_path.replace("gs://", "", 1)
    bucket_name, _, blob_path = path.partition("/")
    input_dir = "/".join(blob_path.split("/")[:-1])
    output_blob = f"{input_dir}/output/{filename}" if input_dir else f"output/{filename}"
    return bucket_name, output_blob


def _upload_to_gcs(data: dict, bucket_name: str, blob_path: str, gcs_output_uri: str):
    """Upload JSON data to a GCS blob. Logs success, permission, and unexpected errors."""
    try:
        from google.cloud import storage

        logger.info(f"Uploading output JSON to GCS: {gcs_output_uri}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(data, ensure_ascii=False, indent=2),
            content_type="application/json",
        )
        logger.info(f"Successfully uploaded output JSON to GCS: {gcs_output_uri}")
        return True
    except ImportError as e:
        logger.error(f"GCS dependencies are not installed: {e}")
        return False
    except Exception as e:
        if "403" in str(e) or "Permission" in str(e):
            logger.error(f"GCS upload permission denied for {gcs_output_uri}: {e}")
        else:
            logger.error(f"GCS upload API error for {gcs_output_uri}: {e}")
        return False


def save_response(feature: str, response: dict, request_id: str | None = None, gcs_file_path: str | None = None):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    filename, job_id = build_filename(feature, request_id)
    local_path = os.path.join(OUTPUT_DIR, filename)

    # Determine output path upfront so it's baked into the JSON before writing
    if gcs_file_path:
        bucket_name, blob_path = _derive_gcs_output_path(gcs_file_path, filename)
        output_path = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"Derived GCS output path: {output_path}")
    else:
        output_path = local_path

    # Update Pub/Sub schema fields
    if "payload" in response:
        response["payload"]["processed_bucket_path"] = output_path

    response["job_id"] = job_id
    response["output_file"] = output_path

    # Always write locally first
    atomic_write_json(response, local_path)
    logger.info(f"Output JSON saved locally: {local_path}")

    # Upload to GCS if source path was provided
    if gcs_file_path:
        success = _upload_to_gcs(response, bucket_name, blob_path, output_path)
        if not success:
            logger.warning(
                f"GCS upload failed — output is only available locally at: {local_path}"
            )

    return response
