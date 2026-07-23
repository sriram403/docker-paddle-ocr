from app.core.logger import get_logger

logger = get_logger("ModelRegistry")

def preload_models(model_type="vl"):
    logger.info(
        "PaddleOCR model uses lazy isolated-worker loading (model_type=%s)",
        model_type,
    )
    return None


def get_paddle_model():
    return None
