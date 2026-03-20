# FILE: app/main.py
from fastapi import FastAPI
from pathlib import Path

from app.core.config import OUTPUT_DIR, ENABLE_PUBSUB_LISTENER
from app.core.logger import get_logger
from app.routes.doc_intel import router as doc_router
from app.core.model_registry import preload_models
from app.core.model_registry import get_paddle_model
from app.routes.health import router as health_router
from app.routes.pubsub_test import router as pubsub_router
from app.services.pubsub_listener import start_listener_thread

logger = get_logger("main")

app = FastAPI(title="JLR Doc Intelligence API")

app.include_router(doc_router)
app.include_router(health_router)
app.include_router(pubsub_router)

@app.on_event("startup")
async def startup_event():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Starting API...")
    preload_models("v3")
    get_paddle_model()
    logger.info(f"Output directory ready at {OUTPUT_DIR}")

    if ENABLE_PUBSUB_LISTENER:
        start_listener_thread()
    else:
        logger.info("Pub/Sub listener disabled (ENABLE_PUBSUB_LISTENER=false)")


@app.get("/health")
def health():
    return {"status": "ok"}