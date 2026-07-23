# FILE: app/main.py
from fastapi import FastAPI
from pathlib import Path

from app.core.config import (
    CHECKPOINT_DIR,
    DEFAULT_MODEL_TYPE,
    ENABLE_PUBSUB_LISTENER,
    LOG_DIR,
    OUTPUT_DIR,
)
from app.core.logger import get_logger
from app.routes.doc_intel import router as doc_router
from app.core.model_registry import preload_models
from app.engine.doc_processor_engine import shutdown_paddle_worker
from app.routes.health import router as health_router

logger = get_logger("main")

app = FastAPI(title="JLR Doc Intelligence API")

app.include_router(doc_router)
app.include_router(health_router)
if ENABLE_PUBSUB_LISTENER:
    from app.routes.pubsub_test import router as pubsub_router

    app.include_router(pubsub_router)

@app.on_event("startup")
async def startup_event():
    for directory in (OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("Starting API...")
    preload_models(DEFAULT_MODEL_TYPE)
    logger.info(f"Output directory ready at {OUTPUT_DIR}")

    if ENABLE_PUBSUB_LISTENER:
        from app.services.pubsub_listener import start_listener_thread

        start_listener_thread()
    else:
        logger.info("Pub/Sub listener disabled (ENABLE_PUBSUB_LISTENER=false)")


@app.on_event("shutdown")
async def shutdown_event():
    shutdown_paddle_worker()
