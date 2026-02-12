from app.engine.doc_processor_engine import load_paddle_model
from app.core.logger import get_logger

logger = get_logger("ModelRegistry")

PADDLE_MODEL = None


def preload_models(model_type="v3"):
    global PADDLE_MODEL

    if PADDLE_MODEL is None:
        logger.info("Preloading PaddleOCR model into GPU memory...")
        PADDLE_MODEL = load_paddle_model(model_type)
        logger.info("PaddleOCR model loaded")

    return PADDLE_MODEL


def get_paddle_model():
    return PADDLE_MODEL
