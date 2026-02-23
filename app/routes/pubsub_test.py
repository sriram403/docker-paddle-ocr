# FILE: app/routes/pubsub_test.py
from fastapi import APIRouter
from google.cloud import pubsub_v1
from google.api_core.exceptions import GoogleAPIError
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger("pubsub_test")

PROJECT_ID = "jlr-dl-iqm"
TOPIC_ID = "iqm_rms_ai"

@router.get("/test-pubsub")
def test_pubsub_connection():
    logger.info("Starting GCP Pub/Sub connectivity test...")
    
    try:
        # Initialize publisher (auto-detects GCP environment credentials)
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
        
        # Message must be a bytestring
        message_data = b"Hello World from JLR Doc Intelligence API!"
        
        logger.info(f"Attempting to publish to topic: {topic_path}")
        
        # Publish the message
        future = publisher.publish(topic_path, message_data)
        
        # Timeout after 10 seconds to prevent the API from hanging
        message_id = future.result(timeout=10)
        
        logger.info(f"SUCCESS: Published message ID {message_id} to {topic_path}")
        
        return {
            "status": "success", 
            "message": "Hello World published successfully",
            "message_id": message_id,
            "topic": topic_path
        }
        
    except GoogleAPIError as e:
        error_msg = f"GCP API Error: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}
        
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}