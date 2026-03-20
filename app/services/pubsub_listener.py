# FILE: app/services/pubsub_listener.py
import json
import base64
import threading
import uuid
from datetime import datetime

from google.cloud import pubsub_v1, storage
from google.api_core.exceptions import GoogleAPIError

from app.core.config import (
    PUBSUB_SUBSCRIPTION,
    PUBSUB_TOPIC,
    AI_PROVIDER,
)
from app.core.logger import get_logger
from app.services.doc_intel_service import DocIntelService
from app.engine.doc_processor_engine import DocumentProcessor, load_paddle_model
from app.storage.output_writer import save_response

logger = get_logger("pubsub_listener")

service = DocIntelService(DocumentProcessor, load_paddle_model)


def _download_from_gcs(gcs_uri: str) -> bytes:
    """Download PDF bytes from a gs://bucket/path URI."""
    # Strip gs:// prefix and split into bucket + blob path
    path = gcs_uri.replace("gs://", "", 1)
    bucket_name, _, blob_path = path.partition("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    logger.info(f"Downloading from GCS: {gcs_uri}")
    return blob.download_as_bytes()


def _publish_event(publisher: pubsub_v1.PublisherClient, event: dict):
    """Publish a Pub/Sub event back to the configured topic."""
    try:
        data = json.dumps(event).encode("utf-8")
        future = publisher.publish(PUBSUB_TOPIC, data)
        message_id = future.result(timeout=10)
        logger.info(f"Published {event.get('event_type')} event, message_id={message_id}")
    except Exception as e:
        logger.error(f"Failed to publish event: {e}")


def _handle_message(message: pubsub_v1.subscriber.message.Message, publisher: pubsub_v1.PublisherClient):
    """Process a single PDFUploaded Pub/Sub message."""
    try:
        raw = base64.b64decode(message.data).decode("utf-8")
        event = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to decode message: {e}")
        message.nack()
        return

    event_type = event.get("event_type")
    if event_type != "PDFUploaded":
        logger.info(f"Ignoring event_type={event_type}, expected PDFUploaded")
        message.ack()
        return

    payload = event.get("payload", {})
    gcs_file_path = payload.get("gcs_file_path")
    regulation_change_id = payload.get("regulation_change_id")
    document_version = payload.get("document_version")

    if not gcs_file_path:
        logger.error("PDFUploaded event missing gcs_file_path — nacking")
        message.nack()
        return

    logger.info(f"Received PDFUploaded | regulation_change_id={regulation_change_id} | path={gcs_file_path}")

    # Publish FileProcessingStarted
    _publish_event(publisher, {
        "event_id": str(uuid.uuid4()),
        "event_type": "FileProcessingStarted",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": "RMS AI",
        "payload": {
            "regulation_change_id": regulation_change_id,
            "document_version": document_version,
            "gcs_file_path": gcs_file_path,
        }
    })

    # Download PDF from GCS
    try:
        pdf_bytes = _download_from_gcs(gcs_file_path)
    except Exception as e:
        logger.error(f"GCS download failed: {e}")
        message.nack()
        return

    # Process document
    response = service.process_document(
        pdf_bytes=pdf_bytes,
        num_pages=10,
        enable_ai_tables=False,
        analysis_model="gpt-4o-mini",
        do_summary=False,
        model_type="v3",
        ai_provider=AI_PROVIDER,
        regulation_change_id=regulation_change_id,
        document_version=document_version,
        gcs_file_path=gcs_file_path,
    )

    # Save output JSON
    response = save_response(
        feature="DOCINTEL",
        response=response,
        request_id=str(regulation_change_id) if regulation_change_id else None,
    )

    # Publish FileProcessingCompleted back to topic
    _publish_event(publisher, response)

    message.ack()
    logger.info(f"Done processing {gcs_file_path}")


def start_pull_listener():
    """
    Blocking function — runs the Pub/Sub pull subscriber loop.
    Intended to be run in a background daemon thread.
    """
    logger.info(f"Starting Pub/Sub pull listener on: {PUBSUB_SUBSCRIPTION}")

    subscriber = pubsub_v1.SubscriberClient()
    publisher = pubsub_v1.PublisherClient()

    def callback(message: pubsub_v1.subscriber.message.Message):
        _handle_message(message, publisher)

    streaming_pull_future = subscriber.subscribe(PUBSUB_SUBSCRIPTION, callback=callback)

    try:
        streaming_pull_future.result()  # blocks until cancelled or error
    except GoogleAPIError as e:
        logger.error(f"Pub/Sub GoogleAPIError: {e}")
        streaming_pull_future.cancel()
    except Exception as e:
        logger.error(f"Pub/Sub listener crashed: {e}")
        streaming_pull_future.cancel()


def start_listener_thread():
    """Start the pull listener in a background daemon thread."""
    t = threading.Thread(target=start_pull_listener, daemon=True, name="pubsub-listener")
    t.start()
    logger.info("Pub/Sub listener thread started")
