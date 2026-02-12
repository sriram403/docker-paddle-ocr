from fastapi import FastAPI
from pathlib import Path

from app.core.config import OUTPUT_DIR
from app.core.logger import get_logger
from app.routes.doc_intel import router as doc_router
from app.core.model_registry import preload_models
from app.core.model_registry import get_paddle_model
from app.routes.health import router as health_router


logger = get_logger("main")

app = FastAPI(title="JLR Doc Intelligence API")

app.include_router(doc_router)
app.include_router(health_router)

@app.on_event("startup")
async def startup_event():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Starting API...")
    preload_models("v3")
    get_paddle_model()
    logger.info(f"Output directory ready at {OUTPUT_DIR}")


@app.get("/health")
def health():
    return {"status": "ok"}
