import asyncio
import logging
import time
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException, Request, Query 
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse 
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded 

# Import our unified predict pipeline
from predict import predict_image 
from config import LOW_CONFIDENCE_THRESHOLD 
from exceptions import PreprocessingError, ModelExecutionError
from api.task_store import TaskStore, TaskResult

logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

# ---------------------------------------------------------------------------
# CORS — explicit origin allowlist instead of wildcard
# ---------------------------------------------------------------------------
_DEFAULT_ORIGINS = "http://localhost:8501,http://localhost:8000"
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("PIXELTRUTH_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------
API_KEY: str = os.getenv("PIXELTRUTH_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Rate limiting — 10 requests per minute per client IP
# ---------------------------------------------------------------------------
RATE_LIMIT: str = os.getenv("PIXELTRUTH_RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)


async def _read_image_bytes(file: UploadFile) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    retry_after = "60"
    try:
        limiter = getattr(request.app.state, "limiter", None)
        view_rate_limit = getattr(request.state, "view_rate_limit", None)
        if limiter and view_rate_limit:
            limit_item, identifiers = view_rate_limit
            stats = limiter.limiter.get_window_stats(limit_item, *identifiers)
            reset_time = stats[0]
            seconds_left = max(1, int(reset_time - time.time()))
            retry_after = str(seconds_left)
        elif exc.limit and hasattr(exc.limit, "limit") and hasattr(exc.limit.limit, "get_expiry"):
            retry_after = str(exc.limit.limit.get_expiry())
    except Exception:
        pass

    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
        headers={"Retry-After": retry_after},
    )


app = FastAPI(
    title="PixelTruth API",
    description="Deepfake detection API that classifies an image as Real or Fake.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Global in-memory task store for async inference.
task_store = TaskStore()


def _format_inference_response(result: dict) -> dict:
    confidence = result["confidence"]
    return {
        "verdict": result["label"],
        "confidence": confidence,
        "is_uncertain": confidence < LOW_CONFIDENCE_THRESHOLD,   # CWE-116/345 fix
        "confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "raw_scores": result["raw"],
        "face_detected": result.get("face_detected", False),
        "face_box": list(result["face_box"]) if result.get("face_box") is not None else None,
    }


def _run_inference_task(task_id: str, image_bytes: bytes, temperature: float = 1.0) -> None:
    task_store.mark_running(task_id)
    try:
        result = predict_image(image_bytes, temperature=temperature)
        task_store.mark_completed(task_id, result)
    except Exception as exc:
        logger.error("Background inference task failed", exc_info=exc)
        task_store.mark_failed(task_id, str(exc))

def _verify_api_key(request: Request) -> None:
    if not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set the X-API-Key header.",
        )


@app.post("/api/detect")
@limiter.limit(RATE_LIMIT)
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    temperature: float = Query(default=1.0, gt=0.0, le=100.0), 
):
    """
    Accepts an uploaded image file and returns deepfake detection results.
    Optional `temperature` parameter controls confidence calibration (default: 1.0).
    """
    _verify_api_key(request)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await _read_image_bytes(file)
        result = await asyncio.to_thread(predict_image, image_bytes, temperature=temperature)
        return _format_inference_response(result)

    except HTTPException:
        raise
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except (ValueError, TypeError) as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ModelExecutionError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model execution.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.post("/api/detect/async", status_code=202)
@limiter.limit(RATE_LIMIT)
async def detect_image_async(
    request: Request,
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    temperature: float = Query(default=1.0, gt=0.0, le=100.0),  
):
    _verify_api_key(request)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await _read_image_bytes(file)
        task_id = task_store.create_task()
        background_tasks.add_task(_run_inference_task, task_id, image_bytes, temperature)
        return {"task_id": task_id}

    except HTTPException:
        raise
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except (ValueError, TypeError) as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ModelExecutionError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model execution.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/api/task/{task_id}", response_model=TaskResult)
async def get_task_status(request: Request, task_id: str):
    _verify_api_key(request)
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task
