from app.core.config import (
    CHECKPOINT_DIR,
    CHECKPOINT_MAX_MB,
    GPU_MEMORY_FRACTION,
    LOG_DIR,
    LOG_MAX_MB,
    OCR_LANGUAGE,
    GCP_PROJECT_ID,
    VLLM_MODEL_NAME,
    VLLM_SERVER_URL,
)
from app.engine.progress import st
import fitz
import json
import os
import hashlib
import uuid
# Must be set BEFORE importing paddle so worker subprocesses inherit it
os.environ["FLAGS_use_eager_mode"] = "1"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = GPU_MEMORY_FRACTION

import gc
import re
import time
from datetime import datetime
import nltk
import openai
import pandas as pd
from collections import defaultdict, Counter
import base64
import logging
import httpx
import unicodedata
import ftfy
import concurrent.futures
import threading
import cv2
import numpy as np
import paddle

# --- CONFIGURATION ---
# Add any PaddleOCR layout classes here that you want to be treated, extracted,
# and downloaded as images (e.g., 'figure', 'image', 'formula', 'chart').
IMAGE_EXTRACTION_CLASSES = [
    "image",
    "figure",
    "formula",
    "equation",
    "chart",
    "diagram",
    "display_formula",
    "table",
    "header_image",
    "footer_image",
    "seal",
]

# Add any classes that might contain the words above but should strictly be treated
# as TEXT instead of extracted as an image (e.g., captions, titles).
TEXT_OVERRIDE_CLASSES = ["figure_title"]

# ── Checkpoint Configuration ───────────────────────────────────────────────
# Folder where mid-run snapshots are stored (created automatically).
# If ALL checkpoint files in CHECKPOINT_DIR collectively exceed this size (MB),
# every checkpoint is deleted and the Resume option is hidden — user must restart.
# Change this value to suit available disk space (e.g. 500, 1000, 2000).
# ──────────────────────────────────────────────────────────────────────────

# ── Alignment Merge Configuration ─────────────────────────────────────────
# Maximum horizontal distance (PDF points) between the left x-coordinate of
# a non-clause block and the nearest preceding clause chunk's left x-coord
# for the block to be merged into that clause chunk (Step 3 of pipeline).
# Increase to merge more aggressively; decrease to keep more blocks standalone.
ALIGNMENT_MERGE_TOLERANCE_PTS = 10
# ──────────────────────────────────────────────────────────────────────────

# ── OCR Language Configuration ────────────────────────────────────────────
# Controls the text recognition model used by PPStructureV3 (non-VL path).
# The model is loaded ONCE per session — this cannot be changed per-page.
# Common values:
#   "en"   → English only (default, fastest)
#   "hi"   → Hindi / Devanagari
#   "ar"   → Arabic
#   "ch"   → Chinese (Simplified)
#   "ja"   → Japanese
#   "ko"   → Korean
#   "ml"   → Multilingual (covers 80+ scripts, ~10% slower than "en")
# Note: The VL model path (PaddleOCRVL) is vision-based and handles any
# script natively regardless of this setting.
# ──────────────────────────────────────────────────────────────────────────

# ── Log Configuration ──────────────────────────────────────────────────────
# Folder where per-session log files are written (created automatically).

# Maximum combined size of all log files before the OLDEST are deleted (MB).
# Deletion continues until the folder is under 50 % of this limit.
# Set to 0 to disable size-based management (keep every log forever).
# ──────────────────────────────────────────────────────────────────────────

# Focused TOC debug trace pages. These logs go to the per-session log file and
# help trace where TOC numbering disappears across the extraction pipeline.
TOC_DEBUG_PAGES = {3, 4, 6, 7}

# ── Module-level checkpoint utilities ────────────────────────────────────────
def _checkpoint_folder_over_limit():
    """
    Returns True (and purges ALL .json files in CHECKPOINT_DIR) if the
    combined size of checkpoint files exceeds CHECKPOINT_MAX_MB.
    """
    if not os.path.isdir(CHECKPOINT_DIR):
        return False
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".json")]
    total_mb = sum(
        os.path.getsize(os.path.join(CHECKPOINT_DIR, f)) for f in files
    ) / (1024 * 1024)
    if total_mb > CHECKPOINT_MAX_MB:
        for f in files:
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, f))
            except Exception:
                pass
        return True
    return False


def _get_checkpoint_path(pdf_bytes_head, num_pages):
    """Derive a stable checkpoint file path from the first 8 KB of the PDF."""
    pdf_hash = hashlib.md5(pdf_bytes_head).hexdigest()[:10]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"ckpt_{pdf_hash}_{num_pages}.json")
# ─────────────────────────────────────────────────────────────────────────────


# ── Module-level log utilities ────────────────────────────────────────────────
def _manage_log_folder():
    """
    Delete the OLDEST log files in LOG_DIR until the total size is below
    50 % of LOG_MAX_MB.  Does nothing when LOG_MAX_MB == 0.
    """
    if LOG_MAX_MB <= 0 or not os.path.isdir(LOG_DIR):
        return
    files = [
        os.path.join(LOG_DIR, f)
        for f in os.listdir(LOG_DIR)
        if f.endswith(".log")
    ]
    if not files:
        return
    total_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
    if total_mb <= LOG_MAX_MB:
        return
    # Sort oldest-modification-time first
    files.sort(key=lambda f: os.path.getmtime(f))
    target_mb = LOG_MAX_MB * 0.5
    while files and total_mb > target_mb:
        oldest = files.pop(0)
        try:
            total_mb -= os.path.getsize(oldest) / (1024 * 1024)
            os.remove(oldest)
        except Exception:
            pass


def _create_session_logger(pdf_name: str) -> logging.Logger:
    """
    Create a dedicated file logger for one processing session.

    The log file is written to  LOG_DIR/<YYYYMMDD_HHMMSS>_<pdf_stem>.log
    A private attribute  _log_path  is attached to the returned Logger so
    the UI can display the path after the run finishes.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    _manage_log_folder()
    _now = datetime.now(tz=__import__('zoneinfo').ZoneInfo("Asia/Kolkata"))
    _hour = _now.strftime("%I").lstrip("0") or "12"
    _min  = _now.strftime("%M")
    _ampm = _now.strftime("%p")
    ts = f"{_now.strftime('%Y%m%d')}_{_hour}_{_min}_{_ampm}"
    safe_name = re.sub(r"[^\w\-.]", "_", os.path.splitext(pdf_name)[0])[:40]
    log_path = os.path.join(LOG_DIR, f"{ts}_{safe_name}.log")
    logger = logging.getLogger(f"session_{ts}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(fh)
    logger._log_path = log_path  # type: ignore[attr-defined]
    return logger
# ─────────────────────────────────────────────────────────────────────────────


try:
    # 1. Force Dynamic Mode (Fixes the int(Tensor) error)
    paddle.disable_static()
    # 2. Try to limit GPU memory fraction if possible (Optional, good for stability)
    # paddle.device.set_device('gpu:0')
except Exception:
    pass

try:
    from paddleocr import PPStructureV3, PaddleOCRVL
except ImportError:
    pass

@st.cache_resource
def download_nltk_data():
    packages = ['punkt', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

download_nltk_data()

@st.cache_resource(max_entries=1)
def load_summary_model():
    """Load the sentence-transformers model once and cache it for the session."""
    try:
        from sentence_transformers import SentenceTransformer
        # Force CPU — GPU is reserved for PaddleOCR
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception:
        return None


@st.cache_resource(max_entries=1)
def load_paddle_model(model_type="vl"):
    try:
        # Force garbage collection before loading
        gc.collect()
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()

        if model_type == "vl":
            return PaddleOCRVL(
    layout_detection_model_name="PP-DocLayoutV3",
    vl_rec_backend="vllm-server",
    vl_rec_server_url=VLLM_SERVER_URL,
    vl_rec_model_name=VLLM_MODEL_NAME,
    vl_rec_max_concurrency=4,
)
        else:
            _rec_model = f"{OCR_LANGUAGE}_PP-OCRv4_mobile_rec" if OCR_LANGUAGE == "en" else f"{OCR_LANGUAGE}_PP-OCRv4_rec"
            return PPStructureV3(
                layout_detection_model_name="PP-DocLayout-L",
                text_recognition_model_name=_rec_model,
                device="gpu", # Change to "cpu" if your computer keeps crashing
                use_table_recognition=True,
                use_doc_orientation_classify=False,
                use_region_detection=True,
                use_doc_unwarping=False
            )
    except Exception as e:
        st.error(f"Failed to load PaddleOCR: {e}")
        return None


# ── Subprocess GPU worker ─────────────────────────────────────────────────────
import multiprocessing as _mp
from app.engine.paddle_worker_process import paddle_worker_process as _paddle_worker_entry


_GPU_LOCK_FILE = "gpu_worker.lock"
_LOCK_STALE_SECONDS = 60   # heartbeat older than this → session is considered dead


def _write_lock(session_id, pdf_name, progress_state):
    """
    Write/refresh the lock file as JSON.

    progress_state is a mutable dict with keys 'page' and 'total' that the
    heartbeat thread reads on every beat so the UI can show live progress.
    """
    try:
        data = {
            "session_id": session_id,
            "started_at": progress_state.get("started_at", ""),
            "pdf": pdf_name,
            "page": progress_state.get("page", 0),
            "total_pages": progress_state.get("total", 0),
            "ts": time.time(),
        }
        # Write atomically via a temp file to avoid partial reads
        _tmp = _GPU_LOCK_FILE + ".tmp"
        with open(_tmp, "w") as _f:
            json.dump(data, _f)
        os.replace(_tmp, _GPU_LOCK_FILE)
    except Exception:
        pass


def _clear_lock():
    """Delete the lock file (called on clean shutdown)."""
    try:
        os.remove(_GPU_LOCK_FILE)
    except Exception:
        pass


def _start_heartbeat(session_id, pdf_name, progress_state):
    """
    Start a daemon thread that refreshes the lock file every 10 seconds.

    Returns (stop_evt, thread) — both must be passed to _stop_heartbeat.
    """
    stop_evt = threading.Event()

    def _beat():
        while not stop_evt.wait(10):
            _write_lock(session_id, pdf_name, progress_state)

    t = threading.Thread(target=_beat, daemon=True, name="gpu_lock_heartbeat")
    t.start()
    return stop_evt, t


def _stop_heartbeat(stop_evt, thread=None):
    """Signal the heartbeat thread to stop, wait for it to finish, then remove the lock.

    Joining the thread before clearing ensures a mid-write heartbeat can't
    re-create the lock file after we delete it (race condition).
    """
    stop_evt.set()
    if thread is not None:
        thread.join(timeout=2)   # wait up to 2 s for the last write to finish
    _clear_lock()


def _check_gpu_lock():
    """
    Inspect the lock file and decide what to do.

    Returns:
        "proceed"        – no lock file, or file is unreadable → safe to run
        "blocked"        – another session has a fresh heartbeat → show wait message
        "stale_cleared"  – lock existed but was stale; it has been removed → safe to run
        dict             – the parsed lock data when returning "blocked"
                           (caller uses it to display progress info to the waiting user)
    """
    if not os.path.exists(_GPU_LOCK_FILE):
        return "proceed"
    try:
        with open(_GPU_LOCK_FILE) as _f:
            data = json.load(_f)
        age = time.time() - data.get("ts", 0)
        if age < _LOCK_STALE_SECONDS:
            return ("blocked", data)   # heartbeat is fresh → session is alive
        # Heartbeat is stale — session crashed or disconnected; clear and proceed
        _clear_lock()
        return "stale_cleared"
    except Exception:
        _clear_lock()
        return "proceed"


def _start_paddle_worker(model_type, logger=None, init_timeout=720):
    """Start a persistent GPU worker subprocess. Returns (proc, task_q, result_q)."""
    ctx = _mp.get_context("spawn")
    task_q  = ctx.Queue()
    result_q = ctx.Queue()
    proc = ctx.Process(
        target=_paddle_worker_entry,
        args=(model_type, OCR_LANGUAGE, task_q, result_q),
        daemon=True,
        name="paddle_gpu_worker",
    )
    proc.start()
    if logger:
        logger.info(f"[GPU WORKER] Subprocess started pid={proc.pid} model={model_type}")

    # Poll in short intervals so a crash (zombie/segfault) is detected within
    # seconds rather than blocking for the full init_timeout.
    deadline = time.time() + init_timeout
    status, data = None, None
    while time.time() < deadline:
        if not proc.is_alive():
            exit_code = proc.exitcode
            proc.kill()
            raise RuntimeError(
                f"GPU worker subprocess crashed during init (exit code {exit_code})"
            )
        try:
            status, data = result_q.get(timeout=5)
            break
        except Exception:
            continue
    else:
        proc.kill()
        raise RuntimeError("GPU worker subprocess failed to start within timeout")

    if status == "init_error":
        proc.kill()
        raise RuntimeError(f"GPU worker init failed: {data}")
    if logger:
        logger.info(f"[GPU WORKER READY] pid={proc.pid}")
    return proc, task_q, result_q


def _stop_paddle_worker(proc, task_q, logger=None):
    """Gracefully shut down the GPU worker subprocess."""
    try:
        task_q.put(None)        # send shutdown signal
        proc.join(timeout=10)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if logger:
        logger.info(f"[GPU WORKER STOPPED] pid={proc.pid}")


def shutdown_paddle_worker():
    """Stop and forget the reusable worker during API shutdown."""
    cached = st.session_state.pop("_paddle_worker", None)
    if cached is not None:
        _stop_paddle_worker(cached["proc"], cached["task_q"])


def _predict_via_worker(proc, task_q, result_q, img_bgr, model_type, logger=None, page_num=None):
    """
    Send an image to the persistent worker for prediction.
    Restarts the worker up to 2 times if it crashes.
    Returns (serialized_results, proc, task_q, result_q).
    Raises RuntimeError if all attempts fail.
    """
    MAX_RESTARTS = 2
    label = f"PAGE {page_num}" if page_num else "PAGE"

    for attempt in range(MAX_RESTARTS + 1):
        # Restart dead worker
        if not proc.is_alive():
            if attempt == MAX_RESTARTS:
                raise RuntimeError(f"[{label}] GPU worker dead after {attempt} restart(s)")
            if logger:
                logger.warning(
                    f"[GPU WORKER RESTART] {label} attempt={attempt + 1}/{MAX_RESTARTS}"
                )
            proc.kill()
            try:
                proc, task_q, result_q = _start_paddle_worker(model_type, logger)
            except Exception as _re:
                if logger:
                    logger.error(f"[GPU WORKER RESTART FAILED] {_re}")
                raise

        try:
            task_q.put((img_bgr.tobytes(), img_bgr.shape))
            status, data = result_q.get(timeout=600)
        except Exception as _te:
            proc.kill()
            if attempt == MAX_RESTARTS:
                raise RuntimeError(f"[{label}] GPU worker timed out / unresponsive: {_te}")
            if logger:
                logger.warning(f"[GPU WORKER TIMEOUT] {label} attempt={attempt + 1}")
            continue

        if status == "ok":
            return data, proc, task_q, result_q

        # Worker reported an error and exited — log it, loop will restart
        if logger:
            logger.warning(
                f"[GPU WORKER ERROR] {label} attempt={attempt + 1}/{MAX_RESTARTS + 1} — {data}"
            )
        if attempt == MAX_RESTARTS:
            raise RuntimeError(f"[{label}] GPU worker failed after {MAX_RESTARTS + 1} attempts: {data}")

    raise RuntimeError(f"[{label}] GPU worker exhausted all retries")
# ─────────────────────────────────────────────────────────────────────────────


class TableExtractor:
    def __init__(self, api_key, model="gpt-4o", provider="openai", session_logger=None):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.session_logger = session_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())

        if self.api_key and self.provider == "openai":
            custom_http_client = httpx.Client(
                http2=False,
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            self.client = openai.OpenAI(
                api_key=self.api_key,
                http_client=custom_http_client,
                max_retries=2
            )
        elif self.provider == "vertex":
            from app.core.ai_client import AIClientFactory

            self.client = AIClientFactory.create("vertex")
        else:
            self.client = None

    def _extract_text_from_gemini_response(self, payload):
        texts = []
        for candidate in payload.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _generate_json_from_image(self, system_prompt, user_text, b64_image):
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content

        if self.provider == "vertex":
            return self.client.generate_content_with_image(
                prompt=user_text,
                base64_image=b64_image,
                system_prompt=system_prompt,
                response_format="json",
                model=self.model,
                temperature=0.0,
            )

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": user_text},
                    {"inlineData": {"mimeType": "image/png", "data": b64_image}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        response = httpx.post(endpoint, params={"key": self.api_key}, json=payload, timeout=60.0)
        response.raise_for_status()
        return self._extract_text_from_gemini_response(response.json())

    def _convert_page_to_image(self, page, dpi=200, crop_rect=None):
        if crop_rect:
            # Add small padding to crop
            crop_rect = crop_rect + (-5, -5, 5, 5)
            # Ensure crop is within page bounds
            crop_rect = crop_rect & page.rect
            pix = page.get_pixmap(dpi=dpi, clip=crop_rect)
        else:
            pix = page.get_pixmap(dpi=dpi)

        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode('utf-8')

    def _check_vector_graphics(self, page):
        h_lines, v_lines = 0, 0
        for path in page.get_drawings():
            for item in path["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2: h_lines += 1
                    elif abs(p1.x - p2.x) < 2: v_lines += 1
                elif item[0] == "re":
                    h_lines += 2; v_lines += 2
        if h_lines > 5 and v_lines > 2: return 50, f"Strong vector graphics (H: {h_lines}, V: {v_lines})"
        elif h_lines > 10 or v_lines > 5: return 25, f"Moderate vector graphics (H: {h_lines}, V: {v_lines})"
        return 0, "No vector graphics found"

    def _check_keywords(self, page):
        text = page.get_text().lower()
        if re.search(r'table of contents|contents', text): return 0, "TOC detected"
        matches = []
        if re.search(r'\btable\s+([ivx\d]+|[\da-z])\b', text): matches.append("Table X")
        if re.search(r'\bannex\s+([ivx\d]+|[\da-z])\b', text): matches.append("Annex X")
        if matches: return 30, f"Keywords: {', '.join(matches)}"
        return 0, "No keywords"

    def _check_header_heuristics(self, page):
        max_caps = 0
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    caps = 0
                    words = []
                    for span in line['spans']: words.extend(span['text'].split())
                    for w in words:
                        if len(w)>2 and w.isupper() and re.match(r'^[A-Z\(\)]+$', w): caps += 1
                    if caps > max_caps: max_caps = caps
        if max_caps >= 4: return 30, f"Strong Headers ({max_caps} CAPS words)"
        elif max_caps >= 2: return 20, f"Moderate Headers ({max_caps} CAPS words)"
        return 0, "No headers"

    def _check_columnar_text(self, page):
        cols = defaultdict(int)
        words = page.get_text("words")
        if not words: return 0, "No text"
        for w in words: cols[round(w[0]/10)] += 1
        sig = len([k for k, c in cols.items() if c > 3])
        if sig >= 5: return 30, f"Strong Columns ({sig})"
        elif sig >= 3: return 20, f"Moderate Columns ({sig})"
        return 0, "No columns"

    def _check_explicit_column_headers(self, page):
        hits = len(re.findall(r'\bcolumn\s+\d+\b', page.get_text(), re.IGNORECASE))
        if hits >= 2: return 200, f"Override: {hits} 'Column X' found"
        elif hits == 1: return 120, "Strong: 'Column X' found"
        return 0, "No explicit headers"

    def _check_fitz_table_finder(self, page):
        try:
            if page.find_tables(strategy="lines").tables: return 30, "Fitz Lines detected"
            if page.find_tables(strategy="text").tables: return 15, "Fitz Text detected"
        except: pass
        return 0, "Fitz found nothing"

    def score_page(self, page, page_num):
        ui_hits = []
        score, reason = self._check_explicit_column_headers(page)
        if score > 0: ui_hits.append("Explicit Headers")

        if score < 200:
            checks = [self._check_fitz_table_finder, self._check_vector_graphics,
                      self._check_keywords, self._check_header_heuristics, self._check_columnar_text]
            for func in checks:
                s, r = func(page)
                score += s
                if s > 0:
                    clean_name = func.__name__.replace("_check_", "").replace("_", " ").title()
                    ui_hits.append(f"{clean_name} ({s})")

        summary_reason = ", ".join(ui_hits) if ui_hits else "No clear table signals"
        return score, summary_reason

    def _extract_page_sync(self, page_data):
        # Unpack tuple: now expects (page, page_num, optional_rect)
        if len(page_data) == 3:
            page, page_num, crop_rect = page_data
        else:
            page, page_num = page_data
            crop_rect = None

        if not self.client: return page_num, []

        try:
            b64_image = self._convert_page_to_image(page, crop_rect=crop_rect)
            system_prompt = """
            You are a data extraction engine. Extract data from images into machine-readable JSON.
            OUTPUT FORMAT: { "tables": [ { "title": "string", "headers": ["s1", "s2"], "rows": [ ["s1", "s2"], ... ] } ] }
            """
            content = self._generate_json_from_image(
                system_prompt,
                "Extract table data exactly as seen.",
                b64_image
            )
            data = json.loads(content)
            processed = []

            for i, t in enumerate(data.get("tables", [])):
                df = pd.DataFrame(t.get("rows", []))
                headers = t.get("headers", [])

                if headers:
                    new_headers = []
                    seen_counts = defaultdict(int)
                    for header in headers:
                        current_count = seen_counts[header]
                        new_headers.append(f"{header}.{current_count}" if current_count > 0 else header)
                        seen_counts[header] += 1

                    if len(new_headers) == df.shape[1]: df.columns = new_headers
                    elif len(new_headers) > df.shape[1]: df.columns = new_headers[:df.shape[1]]
                    else: df.columns = new_headers + [f"Col_{k}" for k in range(df.shape[1]-len(new_headers))]

                processed.append({
                    "page_number": page_num,
                    "table_number": i + 1,
                    "title": t.get("title", f"Table on Page {page_num}"),
                    "html": df.to_html(index=False, border=1, classes="dataframe").replace('\n', ''),
                    "df": df,
                    "b64_img_ref": b64_image
                })
            return page_num, processed
        except Exception:
            return page_num, []

    def process_pages_concurrently(self, doc, pages_to_process, status_callback):
        results = {}
        tasks = []
        for p_num in pages_to_process:
            tasks.append((doc[p_num - 1], p_num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {executor.submit(self._extract_page_sync, task): task[1] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    p_num, tables = future.result()
                    if tables: results[p_num] = tables
                except Exception: pass
                status_callback()
        return results

class DocumentProcessor:
    def __init__(self, ai_provider="openai", document_name="unknown.pdf"):
        self.openai_api_key = None
        self.ai_api_key = None
        self.ai_provider = ai_provider
        self.document_name = document_name
        # Classification batches are independent. Keep a modest concurrency
        # ceiling so labeling is fast without overwhelming provider RPM/TPM
        # limits. All chunk/result mutation still happens on the main thread.
        self.LLM_MAX_CONCURRENCY = 5
        self.LLM_MAX_ATTEMPTS = 3
        self.LLM_RETRY_BASE_SECONDS = 1.0
        self._ai_client_lock = threading.Lock()
        self._ai_client_key = None
        self._openai_client = None
        self._gemini_http_client = None
        self._vertex_client = None
        # Session logger — created in run_pipeline, used throughout processing
        self.session_logger = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())

        # Requirement labels from Labeling_Jaguar.csv. These replace the older
        # regex-driven requirement taxonomy.
        self.REQUIREMENT_LABEL_SPECS = {
            "Limit & Performance": {
                "Definition": "Defines measurable thresholds, minimum/maximum values, pass/fail criteria, or required performance outcomes.",
                "Use_When": "Speed, force, intensity, emissions, braking distance, visibility range, durability, tolerance, response time, efficiency.",
            },
            "Design & Construction": {
                "Definition": "Specifies how a product, component, system, or assembly must be physically designed, built, arranged, or equipped.",
                "Use_When": "Materials, dimensions, layout, mounting, structure, required equipment, component configuration.",
            },
            "Functional & Logic": {
                "Definition": "Describes required system behavior, operating logic, activation/deactivation rules, warnings, interlocks, fail-safe behavior, or software/control functions.",
                "Use_When": "Must activate when, shall not operate unless, warning logic, fallback modes, automated control behavior.",
            },
            "Physical Validation": {
                "Definition": "Requires real-world, bench, laboratory, vehicle-level, or component-level physical testing to demonstrate compliance.",
                "Use_When": "Test rigs, physical test procedures, measurements, endurance tests, impact tests, environmental tests.",
            },
            "Virtual / Simulation": {
                "Definition": "Allows or requires compliance demonstration through simulation, digital models, virtual testing, CAE, software models, or analytical calculation.",
                "Use_When": "Simulation evidence, model validation, virtual test reports, computational analysis.",
            },
            "Marking & Labeling": {
                "Definition": "Requires visible markings, labels, plates, symbols, warnings, inscriptions, identification numbers, or user-facing information on the product/component.",
                "Use_When": "Approval marks, VIN/serial labels, safety warnings, component labels, symbols, packaging labels.",
            },
            "Certification / Admin": {
                "Definition": "Covers approvals, applications, documentation submission, type approval, certificates, declarations, authority communication, and administrative compliance steps.",
                "Use_When": "Application forms, approval authority, certificates, documentation packs, technical files, declarations.",
            },
            "In-Production Testing": {
                "Definition": "Specifies testing or checks during regular manufacturing to ensure produced units continue to conform to the approved type/design.",
                "Use_When": "Production-line tests, sample checks, routine conformity testing, quality checks during manufacture.",
            },
            "Audit & Control": {
                "Definition": "Covers oversight mechanisms, quality systems, inspections, audits, authority checks, manufacturer controls, and conformity of production governance.",
                "Use_When": "COP audits, factory inspection, quality management, authority surveillance, process controls.",
            },
            "In-Service Conformity": {
                "Definition": "Requires verification that products already placed on the road/market continue to comply during real-world use.",
                "Use_When": "In-use testing, market vehicles, service-life compliance, vehicle sampling after sale.",
            },
            "Field Monitoring / Reporting": {
                "Definition": "Requires monitoring, collecting, analysing, or reporting field performance, failures, incidents, defects, or compliance-related data.",
                "Use_When": "Incident reports, defect reporting, warranty/field data, safety monitoring, periodic reports to authorities.",
            },
            "Enforcement Schedule": {
                "Definition": "Defines effective dates, transition timelines, implementation deadlines, phase-in rules, expiry dates, or compliance milestones.",
                "Use_When": "From 1 Jan 2027, after 24 months, transitional provisions, applicability start dates.",
            },
            "Special Cases": {
                "Definition": "Provides exceptions, derogations, alternative treatment, exemptions, limited-use rules, special vehicle/component categories, or conditional applicability.",
                "Use_When": "Exemptions, low-volume vehicles, emergency vehicles, legacy systems, special operating conditions.",
            },
            "Interpretive Clauses": {
                "Definition": "Clarifies how terms, obligations, references, or requirements should be interpreted legally or technically.",
                "Use_When": "Definitions, for the purposes of this regulation, scope interpretation, meaning of terms.",
            },
            "Explanatory Notes": {
                "Definition": "Provides guidance, rationale, examples, explanatory comments, or non-binding clarification that helps understand the requirement but is not itself a direct obligation.",
                "Use_When": "Notes, examples, explanations, rationale, annex guidance, informative text.",
            },
        }
        self.REQUIREMENT_ALLOWED_LABELS = list(self.REQUIREMENT_LABEL_SPECS.keys())
        # Fixed hierarchy agreed in the RMS AI taxonomy review pack. The AI
        # continues to select one of the approved actionable labels above; the
        # application deterministically expands it into the customer-facing
        # R1/R2 fields. A None R2 means the approved label is a unified R1
        # category with no separate action sub-level.
        self.REQUIREMENT_LABEL_LEVEL_MAPPING = {
            "Limit & Performance": ("Core & Technical", "Limit & Performance"),
            "Design & Construction": ("Core & Technical", "Design & Construction"),
            "Functional & Logic": ("Core & Technical", "Functional & Logic"),
            "Physical Validation": ("Testing", "Physical Validation"),
            "Virtual / Simulation": ("Testing", "Virtual / Simulation"),
            "Marking & Labeling": ("Market Access", "Marking & Labeling"),
            "Certification / Admin": ("Market Access", "Certification / Admin"),
            "In-Production Testing": ("Conformity of Production", "In-Production Testing"),
            "Audit & Control": ("Conformity of Production", "Audit & Control"),
            "In-Service Conformity": ("Market Surveillance", "In-Service Conformity"),
            "Field Monitoring / Reporting": ("Market Surveillance", "Field Monitoring / Reporting"),
            "Enforcement Schedule": ("Enforcement Schedule", None),
            "Special Cases": ("Special Cases", None),
            "Interpretive Clauses": ("Interpretive Clauses", None),
            "Explanatory Notes": ("Explanatory Notes", None),
        }
        self.REQUIREMENT_LABEL_GUIDANCE = {
            label: f"{spec['Definition']} Use when: {spec['Use_When']}"
            for label, spec in self.REQUIREMENT_LABEL_SPECS.items()
        }
        self.TEXT_TYPE_ALLOWED_LABELS = [
            "Regulatory",
            "Informative",
            "Comment/Note",
            "Procedural",
            "Definition",
            "Out of Scope",
        ]
        self.CLASSIFICATION_BATCH_SIZE = 12
        self.TOPIC_FALLBACK_LABEL = "Type Approval, CoP, Labels & After-Market"
        self.TOPIC_SPECS = {
            "Braking & Stability": {
                "purpose": "Braking performance, brake systems, stability and stopping behavior.",
                "keywords": ["brake", "braking", "service brake", "parking brake", "emergency brake", "abs", "antilock", "anti-lock", "esc", "stability control", "electronic stability", "vehicle stability", "stopping distance", "aebs", "automatic emergency braking", "brake lining"],
                "title_patterns": [r"\bbrak", r"\babs\b", r"\besc\b", r"\bstability", r"\baebs\b"],
                "clause_patterns": [r"\bstopping\b", r"\bdeceleration\b", r"\bbrake performance\b"],
                "exclusion_patterns": [r"\bsteering column\b", r"\bfield of vision\b", r"\bvisibility\b"]
            },
            "Lighting, Visibility, Windows, Steering": {
                "purpose": "Lighting, signaling, glazing, visibility aids, steering performance and integrity.",
                "keywords": ["lamp", "headlamp", "lighting", "indicator", "stop lamp", "tail lamp", "signal lamp", "mirror", "camera monitor", "field of vision", "windscreen", "windshield", "glazing", "window", "wiper", "demist", "defrost", "steering", "tell-tale", "telltale"],
                "title_patterns": [r"\blamp", r"\bheadlamp", r"\bvisibility\b", r"\bglazing\b", r"\bmirror\b", r"\bwiper\b", r"\bsteering\b"],
                "clause_patterns": [r"\bfield of vision\b", r"\bsteering equipment\b", r"\bsteering control\b"],
                "exclusion_patterns": [r"\bsteering impact\b", r"\bsteering column injury\b"]
            },
            "Crashworthiness, Restraints, VRU, Doors": {
                "purpose": "Occupant crash protection, restraints, vulnerable road users and crash-door retention.",
                "keywords": ["crash protection", "crashworthiness", "occupant protection", "seat belt", "seatbelt", "anchorages", "airbag", "child restraint", "head restraint", "pedestrian protection", "cyclist protection", "door lock", "door latch", "collision", "frontal impact", "side impact", "rear impact"],
                "title_patterns": [r"\bcrash", r"\brestraint", r"\bseat belt", r"\bairbag", r"\bpedestrian", r"\bdoor\b"],
                "clause_patterns": [r"\boccupant\b", r"\bimpact\b", r"\bcollision\b"],
                "exclusion_patterns": [r"\belectric shock\b", r"\bhigh-voltage\b", r"\bthermal runaway\b"]
            },
            "Interior Impact & Ejection Mitigation": {
                "purpose": "Interior injury mitigation and ejection prevention.",
                "keywords": ["interior fitting", "energy absorption", "dashboard", "pillar", "projection", "head impact", "roof crush", "ejection", "ejection mitigation", "curtain airbag", "interior surface"],
                "title_patterns": [r"\binterior\b", r"\bejection\b", r"\bhead impact\b", r"\broof crush\b"],
                "clause_patterns": [r"\benergy absorption\b", r"\boccupant injury\b"],
                "exclusion_patterns": [r"\bchild restraint\b", r"\bhigh-voltage\b"]
            },
            "Post-Crash & EV Safety": {
                "purpose": "Post-impact safety, high-voltage isolation, battery integrity and rescue-related safety.",
                "keywords": ["high-voltage", "electric shock", "post-crash", "post crash", "battery retention", "electrolyte leakage", "thermal runaway", "thermal propagation", "automatic disconnect", "shutdown", "fuel leakage", "hydrogen", "rescue information", "isolation resistance"],
                "title_patterns": [r"\bpost[- ]crash\b", r"\bhigh[- ]voltage\b", r"\bbattery\b", r"\belectric shock\b", r"\bthermal runaway\b"],
                "clause_patterns": [r"\bisolation\b", r"\bafter impact\b", r"\bpost impact\b"],
                "exclusion_patterns": [r"\bstate of health\b", r"\bcapacity retention\b", r"\bdurability\b"]
            },
            "ADAS/AAD, Automated Driving": {
                "purpose": "ADAS and automated driving behavior, sensing, intervention and automation boundaries.",
                "keywords": ["adas", "automated driving", "automation", "automated lane", "lane keeping", "blind spot", "driver assistance", "sensor-based", "operational design domain", "odd", "automation behavior", "lane support"],
                "title_patterns": [r"\badas\b", r"\bautomated\b", r"\blane keep", r"\bblind spot\b", r"\bdriver assistance\b"],
                "clause_patterns": [r"\boperational design\b", r"\bsensing\b", r"\bautomation\b"],
                "exclusion_patterns": [r"\baebs\b", r"\bbrake performance\b", r"\bstopping distance\b"]
            },
            "Driver Monitoring & Event Data Recorders (EDR)": {
                "purpose": "Driver state monitoring, supervision and event data recording.",
                "keywords": ["driver drowsiness", "driver distraction", "driver monitoring", "driver availability", "driver supervision", "edr", "event data recorder", "event data", "data capture trigger", "retained event parameters"],
                "title_patterns": [r"\bdrows", r"\bdistraction\b", r"\bdriver monitoring\b", r"\bedr\b", r"\bevent data\b"],
                "clause_patterns": [r"\bdriver state\b", r"\bdata recorder\b"],
                "exclusion_patterns": [r"\bcybersecurity management\b", r"\bprivacy\b(?!.*driver)"]
            },
            "Criteria Pollutant Control & Fuels": {
                "purpose": "Conventional pollutant emissions, OBD and fuel-specific pollutant control.",
                "keywords": ["nox", "co", "hc", "pm", "pn", "tailpipe", "evaporative emissions", "smoke", "obd", "pollutant", "emission control", "fuel quality"],
                "title_patterns": [r"\bemission", r"\bnox\b", r"\bpm\b", r"\bobd\b", r"\bevaporative\b"],
                "clause_patterns": [r"\btailpipe\b", r"\bpollutant\b", r"\bfuel quality\b"],
                "exclusion_patterns": [r"\bco2\b", r"\bgreenhouse gas\b", r"\bfuel economy\b", r"\benergy consumption\b"]
            },
            "Greenhouse Gases & Fuel Economy": {
                "purpose": "CO2, greenhouse gases, energy use, consumption and fuel economy.",
                "keywords": ["co2", "greenhouse gas", "ghg", "fuel consumption", "fuel economy", "energy consumption", "efficiency", "range", "climate compliance", "fleet target"],
                "title_patterns": [r"\bco2\b", r"\bghg\b", r"\bfuel economy\b", r"\benergy consumption\b", r"\befficiency\b"],
                "clause_patterns": [r"\bfuel consumption\b", r"\bvehicle range\b", r"\bclimate\b"],
                "exclusion_patterns": [r"\bnox\b", r"\bpm\b", r"\bpollutant\b"]
            },
            "Pass-by noise, AVAS": {
                "purpose": "External acoustic noise and AVAS requirements.",
                "keywords": ["pass-by noise", "pass by noise", "stationary noise", "sound level", "acoustic vehicle alerting system", "avas", "minimum sound generation", "external sound emission", "acoustic warning"],
                "title_patterns": [r"\bnoise\b", r"\bavas\b", r"\bacoustic\b", r"\bsound\b"],
                "clause_patterns": [r"\bexternal sound\b", r"\bpass[- ]by\b", r"\bstationary noise\b"],
                "exclusion_patterns": [r"\bemc\b", r"\bradio interference\b", r"\belectromagnetic\b"]
            },
            "Materials, Hazardous Substances & Recycling": {
                "purpose": "Restricted substances, recyclability and end-of-life material obligations.",
                "keywords": ["lead", "mercury", "cadmium", "hexavalent chromium", "recyclability", "recoverability", "end-of-life", "material coding", "hazardous substance", "substance declaration", "recycling"],
                "title_patterns": [r"\bhazardous\b", r"\brecycl", r"\bend[- ]of[- ]life\b", r"\bmaterial\b"],
                "clause_patterns": [r"\bsubstance\b", r"\brecoverability\b"],
                "exclusion_patterns": [r"\bbattery passport\b", r"\bstate of health\b"]
            },
            "EV Battery Durability & Health": {
                "purpose": "Battery ageing, state of health and long-term performance retention.",
                "keywords": ["capacity fade", "state of health", "soh", "durability", "cycle ageing", "capacity retention", "endurance", "battery health", "long-term battery performance"],
                "title_patterns": [r"\bdurability\b", r"\bstate of health\b", r"\bsoh\b", r"\bcapacity retention\b"],
                "clause_patterns": [r"\bageing\b", r"\bcycle\b", r"\bretention threshold\b"],
                "exclusion_patterns": [r"\bpost[- ]crash\b", r"\belectric shock\b", r"\bthermal runaway\b"]
            },
            "Battery Passport regulatory requirements & LCA": {
                "purpose": "Battery passport, traceability, lifecycle data and carbon-footprint declarations.",
                "keywords": ["battery passport", "traceability", "unique battery identification", "carbon footprint", "lifecycle assessment", "lca", "recycled content", "supply-chain", "due diligence", "digital record", "qr code"],
                "title_patterns": [r"\bbattery passport\b", r"\btraceability\b", r"\bcarbon footprint\b", r"\blca\b", r"\blifecycle\b"],
                "clause_patterns": [r"\bsupply chain\b", r"\brecycled content\b", r"\bunique identifier\b"],
                "exclusion_patterns": [r"\bstate of health\b", r"\bpost[- ]crash\b"]
            },
            "Cybersecurity, Software & Data Privacy": {
                "purpose": "Vehicle cybersecurity, software update governance, OTA and privacy.",
                "keywords": ["cybersecurity", "software update", "secure software", "ota", "over-the-air", "privacy", "data protection", "threat", "risk management", "access control", "secure communications", "incident handling", "software version"],
                "title_patterns": [r"\bcybersecurity\b", r"\bsoftware\b", r"\bota\b", r"\bprivacy\b", r"\bdata protection\b"],
                "clause_patterns": [r"\bsecure update\b", r"\bthreat\b", r"\bincident handling\b"],
                "exclusion_patterns": [r"\bimmobilizer\b", r"\bmechanical lock\b", r"\balarm system\b"]
            },
            "Physical Vehicle Security & Anti-Theft": {
                "purpose": "Physical anti-theft devices, immobilizers and unauthorized-entry prevention.",
                "keywords": ["immobilizer", "mechanical lock", "anti-theft", "anti theft", "alarm system", "unauthorized entry", "theft deterrence", "lock cylinder"],
                "title_patterns": [r"\banti[- ]theft\b", r"\bimmobilizer\b", r"\block\b", r"\balarm\b"],
                "clause_patterns": [r"\bunauthorized\b", r"\bentry prevention\b", r"\btheft\b"],
                "exclusion_patterns": [r"\bcybersecurity\b", r"\bsoftware update\b", r"\bota\b"]
            },
            "Electromagnetic Compatibility (EMC), Radio": {
                "purpose": "Electromagnetic emissions, immunity, interference and radio equipment compliance.",
                "keywords": ["emc", "electromagnetic compatibility", "radiated emissions", "conducted emissions", "immunity", "rf interference", "radio equipment", "electronic subassembly", "electromagnetic"],
                "title_patterns": [r"\bemc\b", r"\belectromagnetic\b", r"\bradio\b", r"\brf\b", r"\bimmunity\b"],
                "clause_patterns": [r"\bradiated\b", r"\bconducted\b", r"\binterference\b"],
                "exclusion_patterns": [r"\bpass[- ]by noise\b", r"\bavas\b", r"\bacoustic\b"]
            },
            "Tyres (Safety, Noise, Efficiency), Wheels": {
                "purpose": "Tyres and wheels, including markings, rolling resistance and fitment.",
                "keywords": ["tyre", "tire", "wheel", "rim", "rolling resistance", "wet grip", "tyre noise", "load rating", "speed rating", "fitment", "tyre pressure"],
                "title_patterns": [r"\btyre\b", r"\btire\b", r"\bwheel\b", r"\brim\b"],
                "clause_patterns": [r"\brolling resistance\b", r"\bwet grip\b", r"\bload rating\b"],
                "exclusion_patterns": [r"\bbrake\b", r"\bsuspension\b"]
            },
            "Type Approval, CoP, Labels & After-Market": {
                "purpose": "Procedural approval, conformity of production, labeling and market-surveillance content.",
                "keywords": ["type approval", "approval mark", "certificate", "conformity of production", "cop", "labeling", "marking obligation", "regulatory documentation", "market surveillance", "recall", "replacement part", "after-market", "aftermarket", "approval authority"],
                "title_patterns": [r"\btype approval\b", r"\bconformity of production\b", r"\bcop\b", r"\bmarket surveillance\b", r"\brecall\b", r"\blabel"],
                "clause_patterns": [r"\bapproval mark\b", r"\bcertificate\b", r"\breplacement part\b", r"\bafter[- ]market\b"],
                "exclusion_patterns": []
            },
        }
        self.TOPIC_ALLOWED_LABELS = list(self.TOPIC_SPECS.keys())
        self.TOPIC_LABEL_GUIDANCE = {
            label: spec["purpose"] for label, spec in self.TOPIC_SPECS.items()
        }
        self.TOPIC_BATCH_SIZE = self.CLASSIFICATION_BATCH_SIZE

    def _has_ai(self):
        return bool(self.ai_api_key) or (
            self.ai_provider == "vertex" and bool(GCP_PROJECT_ID)
        )

    def _run_llm_batches_concurrently(self, chunks, batch_size, batch_callable, task_name):
        """Run independent LLM batches concurrently and return successful results."""
        batches = [chunks[start:start + batch_size] for start in range(0, len(chunks), batch_size)]
        if not batches:
            return []

        total_batches = len(batches)
        max_workers = min(self.LLM_MAX_CONCURRENCY, total_batches)
        completed = []
        failed_batches = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(batch_callable, batch, index, total_batches): index
                for index, batch in enumerate(batches, start=1)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    completed.append((index, future.result()))
                except Exception as exc:
                    failed_batches += 1
                    if self.session_logger:
                        self.session_logger.error(
                            f"[LLM BATCH FAILED] task={task_name} batch={index}/{total_batches} "
                            f"error={type(exc).__name__}: {exc}"
                        )

        completed.sort(key=lambda item: item[0])
        if failed_batches:
            st.warning(
                f"{task_name.replace('_', ' ').title()} labeling: "
                f"{failed_batches}/{total_batches} API batch(es) failed after retries."
            )
        return completed

    def _normalize_topic_label(self, label):
        if not label:
            return ""
        text = re.sub(r"\s+", " ", str(label)).strip().casefold()
        for allowed in self.TOPIC_ALLOWED_LABELS:
            if text == allowed.casefold():
                return allowed
        return ""

    def _normalize_yes_no(self, value, default="Yes"):
        text = str(value or "").strip().lower()
        if text in {"yes", "y", "true", "1"}:
            return "Yes"
        if text in {"no", "n", "false", "0"}:
            return "No"
        return default

    def _normalize_requirement_label(self, label):
        if not label:
            return ""
        text = re.sub(r"\s+", " ", str(label)).strip().casefold()
        for allowed in self.REQUIREMENT_ALLOWED_LABELS:
            if text == allowed.casefold():
                return allowed
        return ""

    def _map_requirement_label_levels(self, label):
        """Return the customer-facing (R1, R2) pair for an AI label."""
        normalized = self._normalize_requirement_label(label)
        if not normalized:
            return "", None
        return self.REQUIREMENT_LABEL_LEVEL_MAPPING.get(normalized, ("", None))

    def _normalize_text_type_label(self, label):
        if not label:
            return ""
        text = re.sub(r"\s+", " ", str(label)).strip().casefold()
        for allowed in self.TEXT_TYPE_ALLOWED_LABELS:
            if text == allowed.casefold():
                return allowed
        return ""

    def _topic_fields_default(self):
        return {
            "topic_label": "",
            "secondary_topic_label": "",
            "topic_confidence": 0.0,
            "topic_reason": "",
            "review_required": "Yes",
        }

    def _build_topic_context(self, chunk):
        title = str(chunk.get("title", "") or "")
        clause_id = str(chunk.get("clause_id", "") or "")
        annex = str(chunk.get("annex_appendix", "") or chunk.get("annex_appendix", "") or "")
        refs = " ".join(str(r) for r in chunk.get("references", []) if r)
        content = str(chunk.get("content_verbatim", "") or "")
        page = str(chunk.get("source_page", "") or chunk.get("source_page", "") or "")
        merged = "\n".join(part for part in [title, clause_id, annex, refs, page, content] if part)
        normalized = self._normalize_running_text(merged)
        return {
            "title": title,
            "title_norm": self._normalize_running_text(title),
            "clause_id": clause_id,
            "annex": annex,
            "refs": refs,
            "content": content,
            "content_norm": self._normalize_running_text(content),
            "full_norm": normalized,
        }

    def _is_procedural_chunk(self, chunk, context=None):
        context = context or self._build_topic_context(chunk)
        text = context["full_norm"]
        if not text:
            return True
        procedural_patterns = [
            r"\btype approval\b", r"\bapproval mark\b", r"\bapproval number\b",
            r"\bconformity of production\b", r"\bcop\b", r"\bcertificate\b",
            r"\bmarket surveillance\b", r"\brecall\b", r"\breplacement part\b",
            r"\bafter[- ]market\b", r"\blabel(ing)?\b", r"\btechnical service\b",
            r"\bapproval authority\b", r"\bdocumentation pack\b"
        ]
        hits = sum(1 for pat in procedural_patterns if re.search(pat, text, re.IGNORECASE))
        clause_id = str(chunk.get("clause_id", "") or "").upper()
        if clause_id in {"TOC", "PRELUDE", "NIL"} and len(context["content_norm"].split()) <= 80:
            return True
        return hits >= 2

    def _score_topic_rules(self, chunk, inherited_topic=""):
        context = self._build_topic_context(chunk)
        full_norm = context["full_norm"]
        title_norm = context["title_norm"]
        scores = {}
        reasons = {}

        for label, spec in self.TOPIC_SPECS.items():
            score = 0.0
            hit_reasons = []

            keyword_hits = 0
            for kw in spec.get("keywords", []):
                kw_norm = self._normalize_running_text(kw)
                if kw_norm and kw_norm in full_norm:
                    keyword_hits += 1
            if keyword_hits:
                kw_score = min(0.42, keyword_hits * 0.08)
                score += kw_score
                hit_reasons.append(f"{keyword_hits} keyword hit(s)")

            title_hits = 0
            for pat in spec.get("title_patterns", []):
                if re.search(pat, title_norm, re.IGNORECASE):
                    title_hits += 1
            if title_hits:
                title_score = min(0.34, title_hits * 0.17)
                score += title_score
                hit_reasons.append("title/heading match")

            clause_hits = 0
            for pat in spec.get("clause_patterns", []):
                if re.search(pat, full_norm, re.IGNORECASE):
                    clause_hits += 1
            if clause_hits:
                clause_score = min(0.28, clause_hits * 0.14)
                score += clause_score
                hit_reasons.append("clause-pattern match")

            if inherited_topic == label:
                score += 0.12
                hit_reasons.append("parent-topic carryover")

            exclusion_hits = 0
            for pat in spec.get("exclusion_patterns", []):
                if re.search(pat, full_norm, re.IGNORECASE):
                    exclusion_hits += 1
            if exclusion_hits:
                score -= min(0.4, exclusion_hits * 0.2)
                hit_reasons.append("exclusion penalty")

            scores[label] = round(score, 4)
            reasons[label] = ", ".join(hit_reasons) if hit_reasons else "no strong rule match"

        return context, scores, reasons

    def _pick_topics_from_scores(self, scores):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return "", "", 0.0, 0.0, 0.0
        primary, best = ranked[0]
        second_label, second_score = ("", 0.0)
        if len(ranked) > 1:
            second_label, second_score = ranked[1]
        margin = best - second_score
        confidence = 0.48 + max(0.0, best) * 0.4 + max(0.0, margin) * 0.28
        confidence = max(0.0, min(0.98, confidence))
        secondary = ""
        if second_label and second_score >= 0.34 and (best - second_score) <= 0.18:
            secondary = second_label
        return primary, secondary, round(confidence, 2), round(best, 4), round(second_score, 4)

    def _topic_review_flag(self, confidence):
        return "No" if float(confidence or 0.0) >= 0.85 else "Yes"

    def _apply_topic_values(self, chunk, topic_label, secondary_label="", confidence=0.0, reason="", review_required="Yes"):
        chunk["topic_label"] = self._normalize_topic_label(topic_label) or self.TOPIC_FALLBACK_LABEL
        chunk["secondary_topic_label"] = self._normalize_topic_label(secondary_label)
        if chunk["secondary_topic_label"] == chunk["topic_label"]:
            chunk["secondary_topic_label"] = ""
        try:
            chunk["topic_confidence"] = round(float(confidence), 2)
        except Exception:
            chunk["topic_confidence"] = 0.0
        chunk["topic_reason"] = (reason or "").strip()[:400]
        chunk["review_required"] = self._topic_review_flag(chunk["topic_confidence"])
        return chunk

    def _classify_topics_with_llm(self, chunks, model="gpt-4o"):
        if not self._has_ai() or not chunks:
            return {}

        allowed = "\n".join(f"- {label}: {self.TOPIC_LABEL_GUIDANCE[label]}" for label in self.TOPIC_ALLOWED_LABELS)
        system_prompt = (
            "You classify regulatory PDF chunks into the approved Lexbolt topic buckets.\n"
            "Return valid JSON only.\n"
            "Rules:\n"
            "1. topic_label must be exactly one approved label.\n"
            "2. secondary_topic_label must be blank unless the chunk genuinely spans two areas.\n"
            "3. topic_confidence must be a number from 0 to 1.\n"
            "4. topic_reason must be short and audit-friendly.\n"
            "5. review_required must be Yes or No.\n"
            "6. Do not invent labels outside the approved list."
        )
        results = {}

        def classify_batch(batch, batch_index, total_batches):
            prompt_rows = []
            for chunk in batch:
                prompt_rows.append(
                    f"chunk_id: {chunk.get('chunk_id')}\n"
                    f"clause_id: {chunk.get('clause_id', '')}\n"
                    f"title: {chunk.get('title', '')}\n"
                    f"parent_id: {chunk.get('parent_id', '')}\n"
                    f"annex: {chunk.get('annex_appendix', '')}\n"
                    f"content: {(chunk.get('content_verbatim', '') or '')[:1800]}"
                )
            user_prompt = (
                "APPROVED TOPIC LABELS:\n"
                f"{allowed}\n\n"
                "Return JSON with this exact shape:\n"
                '{"classifications":[{"chunk_id":1,"topic_label":"...","secondary_topic_label":"","topic_confidence":0.91,"topic_reason":"...","review_required":"No"}]}\n\n'
                "CHUNKS:\n"
                + "\n\n---\n\n".join(prompt_rows)
            )
            payload = self._call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                json_mode=True,
                max_tokens=2000,
                request_context=f"topic batch {batch_index}/{total_batches}",
            )
            return self._parse_json_response_text(payload)

        completed_batches = self._run_llm_batches_concurrently(
            chunks, self.TOPIC_BATCH_SIZE, classify_batch, "topic"
        )
        for _, data in completed_batches:
            for item in data.get("classifications", []):
                try:
                    chunk_id = int(item.get("chunk_id"))
                except Exception:
                    continue
                label = self._normalize_topic_label(item.get("topic_label"))
                if not label:
                    continue
                secondary = self._normalize_topic_label(item.get("secondary_topic_label"))
                try:
                    confidence = round(float(item.get("topic_confidence", 0.0)), 2)
                except Exception:
                    confidence = 0.0
                results[chunk_id] = {
                    "topic_label": label,
                    "secondary_topic_label": secondary if secondary != label else "",
                    "topic_confidence": max(0.0, min(1.0, confidence)),
                    "topic_reason": str(item.get("topic_reason", "") or "").strip()[:400],
                    "review_required": self._normalize_yes_no(item.get("review_required"), self._topic_review_flag(confidence)),
                }
        return results

    def _classify_requirement_labels_with_llm(self, chunks, model="gpt-4o"):
        if not self._has_ai() or not chunks:
            return {}

        allowed = "\n".join(
            f"- {label}: {self.REQUIREMENT_LABEL_GUIDANCE[label]}"
            for label in self.REQUIREMENT_ALLOWED_LABELS
        )
        system_prompt = (
            "You classify regulatory PDF chunks into approved requirement-type labels.\n"
            "Return valid JSON only.\n"
            "Rules:\n"
            "1. requirement_label must be exactly one approved label.\n"
            "2. Do not invent labels outside the approved list.\n"
            "3. Use only the chunk text provided.\n"
            "4. Choose the closest approved label using the definition and use-when guidance."
        )
        results = {}

        def classify_batch(batch, batch_index, total_batches):
            prompt_rows = []
            for chunk in batch:
                prompt_rows.append(
                    f"chunk_id: {chunk.get('chunk_id')}\n"
                    f"title: {chunk.get('title', '')}\n"
                    f"clause_id: {chunk.get('clause_id', '')}\n"
                    f"content: {(chunk.get('content_verbatim', '') or '')[:1800]}"
                )

            user_prompt = (
                "APPROVED REQUIREMENT LABELS:\n"
                f"{allowed}\n\n"
                "Return JSON with this exact shape:\n"
                '{"classifications":[{"chunk_id":1,"requirement_label":"..."}]}\n\n'
                "CHUNKS:\n"
                + "\n\n---\n\n".join(prompt_rows)
            )
            payload = self._call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                json_mode=True,
                max_tokens=800,
                request_context=f"requirement batch {batch_index}/{total_batches}",
            )
            return self._parse_json_response_text(payload)

        completed_batches = self._run_llm_batches_concurrently(
            chunks, self.CLASSIFICATION_BATCH_SIZE, classify_batch, "requirement"
        )
        for _, data in completed_batches:
            for item in data.get("classifications", []):
                try:
                    chunk_id = int(item.get("chunk_id"))
                except Exception:
                    continue
                label = self._normalize_requirement_label(item.get("requirement_label"))
                if label:
                    results[chunk_id] = label
        return results

    def assign_requirement_labels(self, chunks, model="gpt-4o", use_llm=True, progress_bar=None):
        if not chunks:
            return chunks, {
                "rule_labeled": 0,
                "llm_labeled": 0,
                "label_distribution": {},
            }

        unresolved = []
        stats = {
            "rule_labeled": 0,
            "llm_labeled": 0,
            "label_distribution": Counter(),
        }
        total_chunks = len(chunks) or 1

        for i, chunk in enumerate(chunks):
            chunk.pop("requirement_label", None)
            chunk.pop("requirement_label_1", None)
            chunk.pop("requirement_label_2", None)
            unresolved.append(chunk)
            if progress_bar is not None:
                progress_bar.progress(
                    min((i + 1) / total_chunks * 0.6, 0.6),
                    text=f"Preparing requirement labels {i + 1}/{total_chunks}"
                )

        if progress_bar is not None and unresolved and use_llm and self._has_ai():
            progress_bar.progress(0.7, text=f"AI-labeling {len(unresolved)} requirement chunk(s)...")

        llm_results = self._classify_requirement_labels_with_llm(unresolved, model=model) if use_llm else {}
        for chunk in unresolved:
            label = self._normalize_requirement_label(llm_results.get(chunk.get("chunk_id")))
            if label:
                label_1, label_2 = self._map_requirement_label_levels(label)
                chunk["requirement_label_1"] = label_1
                chunk["requirement_label_2"] = label_2
                stats["llm_labeled"] += 1

        for chunk in chunks:
            # R2 is the actionable AI label where it exists. For a unified
            # category, R1 itself is the approved AI label and R2 stays null.
            label = self._normalize_requirement_label(
                chunk.get("requirement_label_2") or chunk.get("requirement_label_1")
            )
            if label:
                label_1, label_2 = self._map_requirement_label_levels(label)
                chunk["requirement_label_1"] = label_1
                chunk["requirement_label_2"] = label_2
                stats["label_distribution"][label] += 1

        stats["label_distribution"] = dict(stats["label_distribution"])
        return chunks, stats

    def _classify_text_type_labels_with_llm(self, chunks, model="gpt-4o"):
        if not self._has_ai() or not chunks:
            return {}

        allowed = "\n".join(f"- {label}" for label in self.TEXT_TYPE_ALLOWED_LABELS)
        system_prompt = (
            "You classify regulatory PDF chunks by high-level text nature.\n"
            "Return valid JSON only.\n"
            "Rules:\n"
            "1. ai_label_text_type must be exactly one approved label.\n"
            "2. Use only the provided chunk text.\n"
            "3. Do not invent labels outside the approved list."
        )
        results = {}

        def classify_batch(batch, batch_index, total_batches):
            prompt_rows = []
            for chunk in batch:
                prompt_rows.append(
                    f"chunk_id: {chunk.get('chunk_id')}\n"
                    f"title: {chunk.get('title', '')}\n"
                    f"clause_id: {chunk.get('clause_id', '')}\n"
                    f"content: {(chunk.get('content_verbatim', '') or '')[:1800]}"
                )

            user_prompt = (
                "APPROVED TEXT TYPE LABELS:\n"
                f"{allowed}\n\n"
                "Return JSON with this exact shape:\n"
                '{"classifications":[{"chunk_id":1,"ai_label_text_type":"..."}]}\n\n'
                "CHUNKS:\n"
                + "\n\n---\n\n".join(prompt_rows)
            )
            payload = self._call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                json_mode=True,
                max_tokens=800,
                request_context=f"text_type batch {batch_index}/{total_batches}",
            )
            return self._parse_json_response_text(payload)

        completed_batches = self._run_llm_batches_concurrently(
            chunks, self.CLASSIFICATION_BATCH_SIZE, classify_batch, "text_type"
        )
        for _, data in completed_batches:
            for item in data.get("classifications", []):
                try:
                    chunk_id = int(item.get("chunk_id"))
                except Exception:
                    continue
                label = self._normalize_text_type_label(item.get("ai_label_text_type"))
                if label:
                    results[chunk_id] = label
        return results

    def assign_text_type_labels(self, chunks, model="gpt-4o", progress_bar=None):
        if not chunks or not self._has_ai():
            return chunks

        if progress_bar is not None:
            progress_bar.progress(0.1, text=f"AI-labeling text type for {len(chunks)} chunk(s)...")

        results = self._classify_text_type_labels_with_llm(chunks, model=model)
        total_chunks = len(chunks) or 1
        for i, chunk in enumerate(chunks):
            label = self._normalize_text_type_label(results.get(chunk.get("chunk_id")))
            if label:
                chunk["ai_label_text_type"] = label
            if progress_bar is not None:
                progress_bar.progress(
                    min((i + 1) / total_chunks, 1.0),
                    text=f"Applying text type labels {i + 1}/{total_chunks}"
                )
        return chunks

    def assign_topic_labels(self, chunks, model="gpt-4o", use_llm=True, progress_bar=None):
        if not chunks:
            return chunks, {
                "rule_labeled": 0,
                "llm_labeled": 0,
                "review_required": 0,
                "topic_distribution": {},
            }

        chunk_map = {}
        for chunk in chunks:
            chunk.update(self._topic_fields_default())
            chunk_map[chunk.get("chunk_id")] = chunk

        low_confidence_chunks = []
        stats = {
            "rule_labeled": 0,
            "llm_labeled": 0,
            "review_required": 0,
            "topic_distribution": Counter(),
        }

        total_chunks = len(chunks) or 1
        for i, chunk in enumerate(chunks):
            if progress_bar is not None:
                progress_bar.progress(min((i + 1) / total_chunks * 0.6, 0.6),
                                      text=f"Rule-labeling chunk {i + 1}/{total_chunks}")
            parent = chunk_map.get(chunk.get("parent_id"))
            inherited_topic = ""
            if isinstance(parent, dict) and parent.get("topic_confidence", 0) >= 0.85:
                inherited_topic = parent.get("topic_label", "")

            context, scores, reasons = self._score_topic_rules(chunk, inherited_topic=inherited_topic)
            topic_label, secondary_label, confidence, best_score, second_score = self._pick_topics_from_scores(scores)
            reason = reasons.get(topic_label, "rule-based match") if topic_label else "no strong rule match"

            if best_score <= 0 and self._is_procedural_chunk(chunk, context=context):
                topic_label = self.TOPIC_FALLBACK_LABEL
                secondary_label = ""
                confidence = max(confidence, 0.86)
                reason = "procedural/admin fallback"
            elif not topic_label:
                topic_label = self.TOPIC_FALLBACK_LABEL
                confidence = max(confidence, 0.35)
                reason = "fallback label due to weak topic evidence"

            review_required = self._topic_review_flag(confidence)
            self._apply_topic_values(chunk, topic_label, secondary_label, confidence, reason, review_required)
            stats["rule_labeled"] += 1

            needs_llm = bool(use_llm and self._has_ai()) and (
                chunk["topic_confidence"] < 0.85 or chunk["review_required"] == "Yes"
            )
            if needs_llm:
                low_confidence_chunks.append(chunk)

        if progress_bar is not None and low_confidence_chunks:
            progress_bar.progress(0.65, text=f"AI-labeling {len(low_confidence_chunks)} low-confidence chunk(s)...")
        llm_results = self._classify_topics_with_llm(low_confidence_chunks, model=model) if use_llm else {}
        for chunk in low_confidence_chunks:
            llm_item = llm_results.get(chunk.get("chunk_id"))
            if not llm_item:
                continue
            current_conf = float(chunk.get("topic_confidence", 0.0) or 0.0)
            new_conf = float(llm_item.get("topic_confidence", 0.0) or 0.0)
            if new_conf >= current_conf or chunk.get("review_required") == "Yes":
                self._apply_topic_values(
                    chunk,
                    llm_item.get("topic_label"),
                    llm_item.get("secondary_topic_label", ""),
                    new_conf,
                    llm_item.get("topic_reason", ""),
                    llm_item.get("review_required", self._topic_review_flag(new_conf)),
                )
                stats["llm_labeled"] += 1

        for chunk in chunks:
            topic = self._normalize_topic_label(chunk.get("topic_label")) or self.TOPIC_FALLBACK_LABEL
            chunk["topic_label"] = topic
            chunk["secondary_topic_label"] = self._normalize_topic_label(chunk.get("secondary_topic_label"))
            if chunk["secondary_topic_label"] == topic:
                chunk["secondary_topic_label"] = ""
            chunk["review_required"] = self._normalize_yes_no(chunk.get("review_required"), self._topic_review_flag(chunk.get("topic_confidence", 0.0)))
            if chunk["review_required"] == "Yes":
                stats["review_required"] += 1
            stats["topic_distribution"][topic] += 1

        stats["topic_distribution"] = dict(stats["topic_distribution"])
        return chunks, stats

    def _extract_text_from_gemini_response(self, payload):
        texts = []
        for candidate in payload.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _parse_json_response_text(self, text):
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start_candidates = [idx for idx in [text.find("{"), text.find("[")] if idx != -1]
        if not start_candidates:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        start = min(start_candidates)
        end_object = text.rfind("}")
        end_array = text.rfind("]")
        end = max(end_object, end_array)
        if end == -1 or end <= start:
            raise json.JSONDecodeError("Incomplete JSON object found", text, start)

        return json.loads(text[start:end + 1])

    def _get_openai_client(self):
        """Return one connection-pooled OpenAI client for the active API key."""
        client_key = ("openai", self.ai_api_key)
        with self._ai_client_lock:
            if self._ai_client_key != client_key:
                if self._openai_client is not None:
                    try:
                        self._openai_client.close()
                    except Exception:
                        pass
                if self._gemini_http_client is not None:
                    try:
                        self._gemini_http_client.close()
                    except Exception:
                        pass
                http_client = httpx.Client(
                    timeout=60.0,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                )
                self._openai_client = openai.OpenAI(
                    api_key=self.ai_api_key,
                    http_client=http_client,
                    max_retries=0,
                )
                self._gemini_http_client = None
                self._ai_client_key = client_key
            return self._openai_client

    def _get_gemini_http_client(self):
        """Return one connection-pooled HTTP client for the active Gemini key."""
        client_key = ("gemini", self.ai_api_key)
        with self._ai_client_lock:
            if self._ai_client_key != client_key:
                if self._openai_client is not None:
                    try:
                        self._openai_client.close()
                    except Exception:
                        pass
                if self._gemini_http_client is not None:
                    try:
                        self._gemini_http_client.close()
                    except Exception:
                        pass
                self._openai_client = None
                self._gemini_http_client = httpx.Client(
                    timeout=60.0,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                )
                self._ai_client_key = client_key
            return self._gemini_http_client

    def _is_retryable_llm_error(self, error):
        retryable_openai_errors = tuple(
            error_type for error_type in (
                getattr(openai, "RateLimitError", None),
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "InternalServerError", None),
            )
            if isinstance(error_type, type)
        )
        if retryable_openai_errors and isinstance(error, retryable_openai_errors):
            return True
        if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
            return True
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
        return status_code in (408, 409, 429) or bool(status_code and status_code >= 500)

    def _log_llm_success(self, context, model, elapsed, request_id, usage):
        if not self.session_logger:
            return
        prompt_tokens = completion_tokens = cached_tokens = None
        if usage is not None:
            if isinstance(usage, dict):
                prompt_tokens = usage.get("promptTokenCount")
                completion_tokens = usage.get("candidatesTokenCount")
                cached_tokens = usage.get("cachedContentTokenCount")
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                details = getattr(usage, "prompt_tokens_details", None)
                cached_tokens = getattr(details, "cached_tokens", None) if details else None
        self.session_logger.info(
            f"[LLM DONE] context={context!r} model={model} elapsed={elapsed:.2f}s "
            f"request_id={request_id or '-'} input_tokens={prompt_tokens} "
            f"output_tokens={completion_tokens} cached_tokens={cached_tokens}"
        )

    def _call_llm(self, model, system_prompt, user_prompt, temperature=0.0,
                  json_mode=False, max_tokens=None, request_context="general"):
        if not self._has_ai():
            raise ValueError("Missing AI API key.")

        started_at = time.monotonic()
        for attempt in range(1, self.LLM_MAX_ATTEMPTS + 1):
            try:
                if self.ai_provider == "openai":
                    kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    response = self._get_openai_client().chat.completions.create(**kwargs)
                    self._log_llm_success(
                        request_context,
                        model,
                        time.monotonic() - started_at,
                        getattr(response, "_request_id", None),
                        getattr(response, "usage", None),
                    )
                    return response.choices[0].message.content.strip()

                if self.ai_provider == "vertex":
                    if self._vertex_client is None:
                        from app.core.ai_client import AIClientFactory

                        self._vertex_client = AIClientFactory.create("vertex")
                    result = self._vertex_client.generate_content(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        response_format="json" if json_mode else None,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self._log_llm_success(
                        request_context,
                        model,
                        time.monotonic() - started_at,
                        None,
                        None,
                    )
                    return result.strip()

                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                payload = {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {"temperature": temperature},
                }
                if json_mode:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                    payload["contents"][0]["parts"][0]["text"] = (
                        user_prompt
                        + "\n\nReturn only valid JSON. Do not include markdown fences, commentary, or prose."
                    )
                if max_tokens is not None:
                    payload["generationConfig"]["maxOutputTokens"] = max_tokens

                response = self._get_gemini_http_client().post(
                    endpoint, params={"key": self.ai_api_key}, json=payload
                )
                response.raise_for_status()
                response_json = response.json()
                self._log_llm_success(
                    request_context,
                    model,
                    time.monotonic() - started_at,
                    response.headers.get("x-request-id"),
                    response_json.get("usageMetadata"),
                )
                return self._extract_text_from_gemini_response(response_json)
            except Exception as e:
                retryable = self._is_retryable_llm_error(e)
                if retryable and attempt < self.LLM_MAX_ATTEMPTS:
                    retry_delay = self.LLM_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    if self.session_logger:
                        self.session_logger.warning(
                            f"[LLM RETRY] context={request_context!r} model={model} "
                            f"attempt={attempt}/{self.LLM_MAX_ATTEMPTS} wait={retry_delay:.1f}s "
                            f"error={type(e).__name__}: {e}"
                        )
                    time.sleep(retry_delay)
                    continue

                _emsg = (
                    f"LLM call failed for provider={self.ai_provider} model={model} "
                    f"context={request_context!r} attempt={attempt}: {e}"
                )
                if self.session_logger:
                    self.session_logger.error(f"[LLM ERROR] {_emsg}")
                if request_context == "general":
                    st.error(f"{self.ai_provider.capitalize()} API request failed: {e}")
                raise

    def _get_symbol_map(self):
        return {
            # Greek lowercase
            'a': 'α', 'b': 'β', 'g': 'γ', 'd': 'δ', 'e': 'ε', 'z': 'ζ', 'h': 'η', 'q': 'θ',
            'i': 'ι', 'k': 'κ', 'l': 'λ', 'm': 'μ', 'n': 'ν', 'x': 'ξ', 'o': 'ο', 'p': 'π',
            'r': 'ρ', 's': 'σ', 't': 'τ', 'u': 'υ', 'f': 'φ', 'c': 'χ', 'y': 'ψ', 'w': 'ω',
            # Greek uppercase
            'A': 'Α', 'B': 'Β', 'G': 'Γ', 'D': 'Δ', 'E': 'Ε', 'Z': 'Ζ', 'H': 'Η', 'Q': 'Θ',
            'I': 'Ι', 'K': 'Κ', 'L': 'Λ', 'M': 'Μ', 'N': 'Ν', 'X': 'Ξ', 'O': 'Ο', 'P': 'Π',
            'R': 'Ρ', 'S': 'Σ', 'T': 'Τ', 'U': 'Υ', 'F': 'Φ', 'C': 'Χ', 'Y': 'Ψ', 'W': 'Ω',
            # Math operators (Symbol font byte positions)
            '+': '+', '-': '−', '*': '×', '/': '÷',
            '=': '=', '<': '≤', '>': '≥', '¹': '′', '²': '″',
            '«': '⟨', '»': '⟩', '·': '·', '×': '×', '÷': '÷',
            '\xb4': '′',  # acute accent → prime
            '\xb8': '×',  # cedilla position in Symbol
            '\xd7': '×',  # multiplication sign
            '\xf7': '÷',  # division sign
            '\xb1': '±',  # plus-minus
            '\xb3': '≥',  # in Symbol encoding
            '\xb2': '≤',
            '\xb9': '≠',
            '\xbb': '↔',
            '\xab': '↔',
            '\xae': '←',
            '\xde': '↑',
            '\xaf': '→',
            '\xdf': '↓',
            '\xc5': 'Å',
            '\xb0': '°',
            '°': '°',
            '\xa5': '∞',  # infinity in Symbol
            '\xc0': '≅',
            '\x40': '≅',
            '\xd1': '∇',
            '\xd0': '∂',
            '\xd2': '∫',
            '\xa4': '∃',
            '\x22': '∀',
            '\x24': '∃',
            '\xce': 'Ε',
            '\xcf': '∏',
            '\xd3': 'Σ',
            '\xd6': '√',
        }

    def _calculate_iou(self, boxA, boxB):
        if not boxA or not boxB: return 0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        return interArea / unionArea if unionArea != 0 else 0

    def _get_sorting_key(self, item):
        label = item.get('label', '').lower()
        if 'header' in label: return -1000
        if 'footer' in label: return 10000
        if 'order' in item and item['order'] is not None:
            return item['order']
        bbox = item.get('bbox', [0, 0, 0, 0])
        return int(bbox[1])

    def _assign_synthetic_order(self, blocks):
        """
        Blocks with model-provided `order` integers and blocks without (tables,
        figure_titles, etc.) use different scales: order=1,2,3 vs y-pixels=116+.
        Mixing them causes tables at y=116 to sort AFTER text at order=1.

        Fix: for each block that has no `order`, compute a fractional synthetic
        order by finding which two ordered neighbours (by y-coord) it falls
        between, then placing it between their order values.  This keeps the
        full list in correct top-to-bottom reading order regardless of whether
        the model assigned an explicit order to every block.
        """
        ordered = [(b, float(b['order']), b['bbox'][1])
                   for b in blocks
                   if b.get('order') is not None
                   and b.get('label', '').lower() not in ('header', 'footer')]
        if not ordered:
            return  # nothing to normalise against

        ordered.sort(key=lambda x: x[2])  # sort ordered blocks by y for interpolation

        for blk in blocks:
            label = blk.get('label', '').lower()
            if blk.get('order') is not None or label in ('header', 'footer'):
                continue
            y = blk['bbox'][1]
            # Find the ordered neighbour just above and just below by y-coord
            below = [o for o in ordered if o[2] <= y]
            above = [o for o in ordered if o[2] > y]
            if below and above:
                prev_order, prev_y = below[-1][1], below[-1][2]
                next_order, next_y = above[0][1], above[0][2]
                if next_y > prev_y:
                    frac = (y - prev_y) / (next_y - prev_y)
                else:
                    frac = 0.5
                blk['order'] = prev_order + frac * (next_order - prev_order)
            elif below:
                # Below all ordered blocks → place after the last one
                blk['order'] = below[-1][1] + 0.5
            else:
                # Above all ordered blocks → place before the first one
                blk['order'] = above[0][1] - 0.5

    # ── New Block-First Pipeline Methods ─────────────────────────────────────

    def _ocr_and_create_block_chunks(
        self, valid_blocks, page, p_num, viz_image, img_bgr,
        scale_x, scale_y, header_cutoff, footer_cutoff,
        repeated_headers, repeated_footers,
        table_counter, image_counter, page_image_bboxes, metrics,
    ):
        """
        STEP 1: Convert each PaddleOCR layout block into one initial chunk.
        Returns (block_chunks, table_counter, image_counter, page_image_bboxes).
        All chunks start with clause_id=None, parent_id=None, level=None,
        appendix/annex=None so that later passes can annotate them cleanly.
        """
        block_chunks = []
        prev_block_label = ''

        for block in valid_blocks:
            label = block['label'].lower()
            bbox  = block['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # _forced_zone is set by the zone-tagger above for unlabeled text
            # blocks that sit inside a geometric header/footer margin.
            forced_zone = block.get('_forced_zone')  # 'header', 'footer', or None

            # Edge blocks: explicitly labeled OR geometrically zoned.
            is_edge_block = label in ('header', 'footer', 'footnote', 'footnotes') or bool(forced_zone)
            # Canonical text_type for edge blocks ('footnotes' → 'footnote').
            effective_edge_type = forced_zone or (
                'footnote' if label in ('footnote', 'footnotes') else label
            )

            is_img_block = (
                any(cls in label for cls in IMAGE_EXTRACTION_CLASSES)
                and not any(neg in label for neg in TEXT_OVERRIDE_CLASSES)
            )

            # ── viz drawing ────────────────────────────────────────────────
            if 'table' in label:                color = (0, 0, 255)
            elif is_img_block:                  color = (0, 165, 255)
            elif is_edge_block:                 color = (128, 0, 128)
            elif label == 'vision_footnote':    color = (255, 128, 0)
            elif 'title' in label:              color = (255, 0, 0)
            else:                               color = (0, 255, 0)
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
            _order_prefix = f"{block['order']} " if block.get('order') is not None else ""
            cv2.putText(viz_image, f"{_order_prefix}{label.upper()[:18]}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            pdf_rect = fitz.Rect(
                bbox[0] * scale_x, bbox[1] * scale_y,
                bbox[2] * scale_x, bbox[3] * scale_y,
            )
            blk_bbox = [
                round(pdf_rect.x0, 1), round(pdf_rect.y0, 1),
                round(pdf_rect.x1, 1), round(pdf_rect.y1, 1),
            ]

            # ── columnar-content heuristic (unchanged from old pipeline) ───
            if (label == 'content'
                    and prev_block_label in ('paragraph_title', 'doc_title', 'title')
                    and (bbox[3] - bbox[1]) > 80):
                metrics.setdefault('detected_columnar_content',
                                   defaultdict(list))[p_num].append(pdf_rect)

            # ── HEADER / FOOTER / FOOTNOTE block ──────────────────────────
            # Extract as a text chunk tagged with the appropriate text_type.
            # Repeated-line stripping is NOT applied here because these blocks
            # ARE the page chrome — their content is intentionally kept as-is.
            # If fitz finds no text, fall back to an image crop (no zone skip).
            if is_edge_block and not is_img_block:
                block_dict = page.get_text("dict", clip=pdf_rect)
                all_line_texts   = []
                all_line_words   = []
                first_line_rich  = []
                second_line_rich = []
                fitz_lines_found = False

                prev_x1   = None
                prev_size = 12.0

                for b in block_dict.get("blocks", []):
                    if "lines" not in b:
                        continue
                    for l in b["lines"]:
                        line_text_parts = []
                        line_words      = []
                        line_rich       = []
                        prev_x1   = None
                        prev_size = 12.0

                        for s in l["spans"]:
                            txt = self._clean_text(s["text"], s["font"])
                            if not txt.strip():
                                if s.get("bbox"):
                                    prev_x1   = s["bbox"][2]
                                    prev_size = s.get("size", prev_size)
                                continue
                            is_bold = self._is_span_bold(s)
                            line_rich.append({"text": txt, "is_bold": is_bold})
                            line_words.append({
                                "page": p_num, "text": txt,
                                "bbox": [round(x) for x in s["bbox"]],
                            })
                            if line_text_parts and prev_x1 is not None and s.get("bbox"):
                                gap      = s["bbox"][0] - prev_x1
                                font_sz  = s.get("size", prev_size)
                                sep      = " " if gap >= font_sz * 0.25 else ""
                                line_text_parts.append(sep + txt)
                            else:
                                line_text_parts.append(txt)
                            prev_x1   = s["bbox"][2] if s.get("bbox") else prev_x1
                            prev_size = s.get("size", prev_size)

                        full_line = "".join(line_text_parts).strip()
                        if not full_line:
                            continue
                        all_line_texts.append(full_line)
                        all_line_words.extend(line_words)
                        if not fitz_lines_found:
                            first_line_rich = line_rich
                        elif len(all_line_texts) == 2:
                            second_line_rich = line_rich
                        fitz_lines_found = True

                if not fitz_lines_found:
                    ocr_text = block.get('ocr_text', '').strip()
                    if ocr_text:
                        all_line_texts   = [ocr_text]
                        all_line_words   = []
                        fitz_lines_found = True
                    else:
                        # Rasterize fallback — no zone exclusion for edge blocks
                        page_image_bboxes.append(bbox)
                        image_counter += 1

                if fitz_lines_found:
                    _edge_sep = " " if effective_edge_type in ('header', 'footer', 'footnote', 'vision_footnote') else "\n"
                    chunk = {
                        'chunk_id': None, 'clause_id': None, 'parent_id': None,
                        'level': None, 'annex_appendix': None, 'title': "NIL",
                        'content_verbatim': _edge_sep.join(all_line_texts),
                        'source_page': str(p_num),
                        'content_words': all_line_words,
                        'text_type': effective_edge_type,
                        'block_bboxes': {str(p_num): [blk_bbox]},
                        '_first_line_rich_spans': first_line_rich,
                        '_second_line_rich_spans': second_line_rich,
                        '_block_left_x': blk_bbox[0],
                        '_block_right_x': blk_bbox[2],
                        '_page_width': round(page.rect.width, 1),
                        '_block_centered': abs((blk_bbox[0] + blk_bbox[2]) / 2 - page.rect.width / 2) < page.rect.width * 0.15,
                        '_block_label': label,
                    }
                    block_chunks.append(chunk)

            # ── TABLE block ────────────────────────────────────────────────
            elif 'table' in label:
                metrics['detected_tables'][p_num].append(pdf_rect)

                # ── Misclassification rescue ───────────────────────────────
                # The vision model occasionally labels a clause-list page as a
                # table.  Before committing to text_type='table' (which skips
                # all clause extraction), extract the fitz text and check
                # whether multiple lines begin with a clause-number pattern
                # (e.g. "86.116–94   Calibrations…").  If so, the block is
                # almost certainly a clause list; promote it to text_type='text'
                # so the normal heading / clause-id pipeline can process it.
                _CLAUSE_LINE_RE = re.compile(
                    r'^\s*\d{1,3}\.\d{1,3}(?:\.\d+)*(?:[–\-—]\d{2,4})?\b'
                )
                _tbl_block_dict = page.get_text("dict", clip=pdf_rect)
                _tbl_line_texts, _tbl_line_words = [], []
                _tbl_first_rich, _tbl_second_rich = [], []
                _tbl_fitz_found = False
                _prev_x1_t, _prev_sz_t = None, 12.0

                for _b in _tbl_block_dict.get("blocks", []):
                    if "lines" not in _b:
                        continue
                    for _l in _b["lines"]:
                        _lparts, _lwords, _lrich = [], [], []
                        _prev_x1_t, _prev_sz_t = None, 12.0
                        for _s in _l["spans"]:
                            _stxt = self._clean_text(_s["text"], _s["font"])
                            if not _stxt.strip():
                                if _s.get("bbox"):
                                    _prev_x1_t = _s["bbox"][2]
                                    _prev_sz_t = _s.get("size", _prev_sz_t)
                                continue
                            _lrich.append({"text": _stxt, "is_bold": self._is_span_bold(_s)})
                            _lwords.append({"page": p_num, "text": _stxt,
                                            "bbox": [round(x) for x in _s["bbox"]]})
                            if _lparts and _prev_x1_t is not None and _s.get("bbox"):
                                _gap = _s["bbox"][0] - _prev_x1_t
                                _sep = " " if _gap >= _s.get("size", _prev_sz_t) * 0.25 else ""
                                _lparts.append(_sep + _stxt)
                            else:
                                _lparts.append(_stxt)
                            _prev_x1_t = _s["bbox"][2] if _s.get("bbox") else _prev_x1_t
                            _prev_sz_t = _s.get("size", _prev_sz_t)
                        _full = "".join(_lparts).strip()
                        if not _full:
                            continue
                        _tbl_line_texts.append(_full)
                        _tbl_line_words.extend(_lwords)
                        if not _tbl_fitz_found:
                            _tbl_first_rich = _lrich
                        elif len(_tbl_line_texts) == 2:
                            _tbl_second_rich = _lrich
                        _tbl_fitz_found = True

                _clause_hits = sum(
                    1 for ln in _tbl_line_texts if _CLAUSE_LINE_RE.match(ln)
                )
                _is_clause_list = _tbl_fitz_found and _clause_hits >= 3

                if _is_clause_list:
                    # Rescue: treat as plain text so clause extraction runs
                    chunk = {
                        'chunk_id': None, 'clause_id': None, 'parent_id': None,
                        'level': None, 'annex_appendix': None, 'title': "NIL",
                        'content_verbatim': "\n".join(_tbl_line_texts),
                        'source_page': str(p_num),
                        'content_words': _tbl_line_words,
                        'text_type': 'text',
                        'block_bboxes': {str(p_num): [blk_bbox]},
                        '_first_line_rich_spans': _tbl_first_rich,
                        '_second_line_rich_spans': _tbl_second_rich,
                        '_block_left_x': blk_bbox[0],
                        '_block_right_x': blk_bbox[2],
                        '_page_width': round(page.rect.width, 1),
                        '_block_centered': abs((blk_bbox[0] + blk_bbox[2]) / 2 - page.rect.width / 2) < page.rect.width * 0.15,
                        '_block_label': label,
                    }
                    block_chunks.append(chunk)
                else:
                    chunk = {
                        'chunk_id': None, 'clause_id': None, 'parent_id': None,
                        'level': None, 'annex_appendix': None, 'title': "NIL",
                        'content_verbatim': '',
                        'source_page': str(p_num), 'content_words': [],
                        'text_type': 'table',
                        'block_bboxes': {str(p_num): [blk_bbox]},
                        '_first_line_rich_spans': [],
                        '_block_left_x': blk_bbox[0],
                        '_block_right_x': blk_bbox[2],
                        '_page_width': round(page.rect.width, 1),
                        '_block_centered': abs((blk_bbox[0] + blk_bbox[2]) / 2 - page.rect.width / 2) < page.rect.width * 0.15,
                        '_block_label': label,
                    }
                    block_chunks.append(chunk)
                    table_counter += 1

            # ── VISION FOOTNOTE block ─────────────────────────────────────
            # A caption/footnote element emitted by the model for a table or
            # figure.  Extract text like a regular block but stamp it with
            # text_type='vision_footnote'.  parent_id is linked to the nearest
            # preceding table/figure-bearing chunk in a later pass.
            elif label == 'vision_footnote':
                block_dict = page.get_text("dict", clip=pdf_rect)
                all_line_texts   = []
                all_line_words   = []
                first_line_rich  = []
                second_line_rich = []
                fitz_lines_found = False
                prev_x1   = None
                prev_size = 12.0

                for b in block_dict.get("blocks", []):
                    if "lines" not in b:
                        continue
                    for l in b["lines"]:
                        line_text_parts = []
                        line_words      = []
                        line_rich       = []
                        prev_x1   = None
                        prev_size = 12.0

                        for s in l["spans"]:
                            txt = self._clean_text(s["text"], s["font"])
                            if not txt.strip():
                                if s.get("bbox"):
                                    prev_x1   = s["bbox"][2]
                                    prev_size = s.get("size", prev_size)
                                continue
                            is_bold = self._is_span_bold(s)
                            line_rich.append({"text": txt, "is_bold": is_bold})
                            line_words.append({
                                "page": p_num, "text": txt,
                                "bbox": [round(x) for x in s["bbox"]],
                            })
                            if line_text_parts and prev_x1 is not None and s.get("bbox"):
                                gap      = s["bbox"][0] - prev_x1
                                font_sz  = s.get("size", prev_size)
                                sep      = " " if gap >= font_sz * 0.25 else ""
                                line_text_parts.append(sep + txt)
                            else:
                                line_text_parts.append(txt)
                            prev_x1   = s["bbox"][2] if s.get("bbox") else prev_x1
                            prev_size = s.get("size", prev_size)

                        full_line = "".join(line_text_parts).strip()
                        if not full_line:
                            continue
                        all_line_texts.append(full_line)
                        all_line_words.extend(line_words)
                        if not fitz_lines_found:
                            first_line_rich = line_rich
                        elif len(all_line_texts) == 2:
                            second_line_rich = line_rich
                        fitz_lines_found = True

                if not fitz_lines_found:
                    ocr_text = block.get('ocr_text', '').strip()
                    if ocr_text:
                        all_line_texts   = [ocr_text]
                        all_line_words   = []
                        fitz_lines_found = True

                if fitz_lines_found:
                    chunk = {
                        'chunk_id': None, 'clause_id': None, 'parent_id': None,
                        'level': None, 'annex_appendix': None, 'title': "NIL",
                        'content_verbatim': " ".join(all_line_texts),
                        'source_page': str(p_num),
                        'content_words': all_line_words,
                        'text_type': 'vision_footnote',
                        'block_bboxes': {str(p_num): [blk_bbox]},
                        '_first_line_rich_spans': first_line_rich,
                        '_second_line_rich_spans': second_line_rich,
                        '_block_left_x': blk_bbox[0],
                        '_block_right_x': blk_bbox[2],
                        '_page_width': round(page.rect.width, 1),
                        '_block_centered': abs((blk_bbox[0] + blk_bbox[2]) / 2 - page.rect.width / 2) < page.rect.width * 0.15,
                        '_block_label': label,
                    }
                    block_chunks.append(chunk)

            # ── IMAGE / FIGURE / FORMULA / HEADER_IMAGE / FOOTER_IMAGE / SEAL ──
            # Collect bbox for cropping only — no standalone chunk.
            # The base64 crop is attached to the nearest chunk by the
            # Y-position attachment pass at the end of the pipeline.
            elif is_img_block:
                page_image_bboxes.append(bbox)
                image_counter += 1

            # ── TEXT / HEADING block ───────────────────────────────────────
            else:
                clip_rect = pdf_rect
                if label in ('doc_title', 'title', 'paragraph_title'):
                    clip_rect = fitz.Rect(
                        pdf_rect.x0, pdf_rect.y0, pdf_rect.x1, pdf_rect.y1 + 6
                    )

                block_dict = page.get_text("dict", clip=clip_rect)
                all_line_texts   = []
                all_line_words   = []
                first_line_rich  = []
                second_line_rich = []
                fitz_lines_found = False

                prev_x1   = None
                prev_size = 12.0

                for b in block_dict.get("blocks", []):
                    if "lines" not in b:
                        continue
                    for l in b["lines"]:
                        line_text_parts = []
                        line_words      = []
                        line_rich       = []
                        prev_x1   = None
                        prev_size = 12.0

                        for s in l["spans"]:
                            txt = self._clean_text(s["text"], s["font"])
                            if not txt.strip():
                                if s.get("bbox"):
                                    prev_x1   = s["bbox"][2]
                                    prev_size = s.get("size", prev_size)
                                continue
                            is_bold = self._is_span_bold(s)
                            line_rich.append({"text": txt, "is_bold": is_bold})
                            line_words.append({
                                "page": p_num, "text": txt,
                                "bbox": [round(x) for x in s["bbox"]],
                            })
                            if line_text_parts and prev_x1 is not None and s.get("bbox"):
                                gap      = s["bbox"][0] - prev_x1
                                font_sz  = s.get("size", prev_size)
                                sep      = " " if gap >= font_sz * 0.25 else ""
                                line_text_parts.append(sep + txt)
                            else:
                                line_text_parts.append(txt)
                            prev_x1   = s["bbox"][2] if s.get("bbox") else prev_x1
                            prev_size = s.get("size", prev_size)

                        full_line = "".join(line_text_parts).strip()
                        if not full_line:
                            continue

                        # header/footer filtering ──────────────────────────
                        full_line, stripped_footer = self._strip_repeated_footer_suffix(
                            full_line, repeated_footers)
                        if stripped_footer:
                            metrics['removed_footer_lines'] += 1
                            metrics['dropped_words'] += len(stripped_footer.split())
                            ns = self._normalize_edge_comparison_text(stripped_footer)
                            if ns and ns not in metrics['removed_edge_line_samples']:
                                if len(metrics['removed_edge_line_samples']) < 10:
                                    metrics['removed_edge_line_samples'].append(ns)

                        if full_line.strip() and self._is_safe_repeated_edge_line(
                                full_line, line_words, header_cutoff, footer_cutoff,
                                repeated_headers, repeated_footers):
                            y_center = self._line_center_from_words(line_words)
                            if y_center is not None and y_center <= header_cutoff:
                                metrics['removed_header_lines'] += 1
                            else:
                                metrics['removed_footer_lines'] += 1
                            metrics['dropped_words'] += len(full_line.split())
                            ns = self._normalize_edge_comparison_text(full_line)
                            if ns and ns not in metrics['removed_edge_line_samples']:
                                if len(metrics['removed_edge_line_samples']) < 10:
                                    metrics['removed_edge_line_samples'].append(ns)
                            continue

                        if full_line.strip():
                            all_line_texts.append(full_line)
                            all_line_words.extend(line_words)
                            metrics['kept_words'] += len(full_line.split())
                            if not fitz_lines_found:
                                first_line_rich = line_rich   # first non-empty line
                            elif len(all_line_texts) == 2:
                                second_line_rich = line_rich  # second non-empty line
                            fitz_lines_found = True

                if not fitz_lines_found:
                    ocr_text = block.get('ocr_text', '').strip()
                    if ocr_text:
                        all_line_texts   = [ocr_text]
                        all_line_words   = []
                        fitz_lines_found = True
                    else:
                        # rasterize fallback — skip if in footer/header zone
                        if not (pdf_rect.y0 >= footer_cutoff or pdf_rect.y1 <= header_cutoff):
                            page_image_bboxes.append(bbox)
                else:
                    all_line_texts = self._restore_missing_structural_prefix(
                        all_line_texts,
                        block.get('ocr_text', ''),
                    )

                if fitz_lines_found:
                    chunk = {
                        'chunk_id': None, 'clause_id': None, 'parent_id': None,
                        'level': None, 'annex_appendix': None, 'title': "NIL",
                        'content_verbatim': "\n".join(all_line_texts),
                        'source_page': str(p_num),
                        'content_words': all_line_words,
                        'text_type': 'text',
                        'block_bboxes': {str(p_num): [blk_bbox]},
                        '_first_line_rich_spans': first_line_rich,
                        '_second_line_rich_spans': second_line_rich,
                        '_block_left_x': blk_bbox[0],
                        '_block_right_x': blk_bbox[2],
                        '_page_width': round(page.rect.width, 1),
                        '_block_centered': abs((blk_bbox[0] + blk_bbox[2]) / 2 - page.rect.width / 2) < page.rect.width * 0.15,
                        '_block_label': label,
                    }
                    block_chunks.append(chunk)

            prev_block_label = label

        return block_chunks, table_counter, image_counter, page_image_bboxes

    # ─────────────────────────────────────────────────────────────────────────

    def _annotate_clause_ids(self, raw_chunks, regex_map, heading_priority, month_pattern):
        """
        STEP 2: Separate regex annotation pass over all initial block chunks.
        Populates clause_id, Title, level, text_type on matching chunks.
        Non-matching chunks keep clause_id=None.
        """
        # ── Pass 1: per-chunk regex annotation ───────────────────────────────
        _toc_anchor_found = False   # ensure only the first "Contents" header is promoted
        for chunk in raw_chunks:
            # PaddleOCR occasionally mis-labels a numbered section heading
            # (e.g. "1.\nLEGISLATIVE PROVISIONS") as a running 'header' block.
            # Detect this pattern — digit(s) alone on the first line followed by
            # an ALL-CAPS title on the second line — and re-label as 'text' so
            # normal clause_id annotation applies below.
            if chunk.get('text_type') == 'header':
                _cv_lines = [ln.strip() for ln in chunk.get('content_verbatim', '').splitlines() if ln.strip()]
                if len(_cv_lines) >= 2:
                    _n = re.match(r'^(\d{1,3}(?:\.\d+)*)\.?\s*$', _cv_lines[0])
                    if _n and re.match(r'^[A-Z][A-Z\s\(\)]{2,}$', _cv_lines[1]):
                        chunk['text_type'] = 'text'

            if chunk.get('text_type') in ('table', 'figure', 'header', 'footer', 'footnote', 'vision_footnote'):
                # Exception: a page-header block whose text is "Contents" or
                # "Table of Contents" is the TOC section title, not a running
                # header.  Promote it to the TOC anchor so that Pass 2 can
                # enter in_toc mode and correctly tag the entries below it.
                #
                # Only do this ONCE (the first match in document order).
                # Appendix A or embedded regulations may contain their own
                # "Contents" headings; promoting those would create a second
                # TOC anchor and accidentally pull in Appendix content.
                if chunk.get('text_type') in ('header', 'prelude') and not _toc_anchor_found:
                    _hdr_content = chunk.get('content_verbatim', '')
                    _hdr_first   = next(
                        (ln.strip() for ln in _hdr_content.splitlines() if ln.strip()), ''
                    )
                    _hdr_first = re.sub(r'[\x00-\x1f\x7f]', '', _hdr_first)
                    if _hdr_first and regex_map['toc_start_heading'].match(_hdr_first):
                        chunk['clause_id'] = "TOC"
                        chunk['title']     = "Table of Contents"
                        chunk['text_type'] = 'toc'
                        chunk['level']     = 0
                        _toc_anchor_found  = True
                continue

            content    = chunk['content_verbatim']
            first_line = ""
            for ln in content.splitlines():
                if ln.strip():
                    first_line = ln.strip()
                    break
            first_line = re.sub(r'[\x00-\x1f\x7f]', '', first_line)
            if not first_line:
                continue

            rich_spans  = chunk.get('_first_line_rich_spans', [])
            match_found = False
            found_type  = None
            match_obj   = None

            for h_type in heading_priority:
                m = regex_map[h_type].match(first_line)
                if m:
                    if h_type == 'content_start_heading' and not m.group(1).isupper():
                        continue
                    if h_type == 'roman_upper_heading' and not m.group(1).isupper():
                        continue
                    if h_type == 'roman_upper_alpha_heading' and not m.group(1)[:-1].isupper():
                        continue
                    if h_type == 'preamble_heading' and not m.group(1).isupper():
                        continue
                    if h_type in ['section_general_heading', 'chapter_heading']:
                        if re.search(r'(below|above)\.?\s*$', first_line, re.IGNORECASE):
                            continue
                        # Reject "Section 86.107-98 includes text..." — a CFR-style
                        # number (digits.digits) after SECTION is a cross-reference
                        # in running text, not a real section heading title.
                        if h_type == 'section_general_heading':
                            _sec_id = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else ''
                            if re.match(r'^\d+\.\d+', _sec_id):
                                continue
                    match_found = True
                    found_type  = h_type
                    match_obj   = m
                    break

            if not match_found:
                # Fallback: scan all lines for a TOC section marker.
                # Handles title blocks where "CONTENTS" is not the first line
                # (e.g. "REGULATION NO. 79\n...\nCONTENTS").
                if not _toc_anchor_found:
                    _toc_re = regex_map.get('toc_start_heading')
                    if _toc_re:
                        for _toc_ln in content.splitlines():
                            _toc_ln_clean = re.sub(r'[\x00-\x1f\x7f]', '', _toc_ln.strip())
                            if _toc_ln_clean and _toc_re.match(_toc_ln_clean):
                                chunk['clause_id'] = "TOC"
                                chunk['title']     = "Table of Contents"
                                chunk['text_type'] = 'toc'
                                chunk['level']     = 0
                                _toc_anchor_found  = True
                                break
                continue

            raw_id = "NIL"
            if match_obj.groups():
                if found_type == 'appendix_heading':
                    raw_id = f"{match_obj.group(1).capitalize()} {match_obj.group(2).upper()}"
                else:
                    raw_id = match_obj.group(1).strip().strip('.')
            raw_id = self._normalize_heading_clause_id(raw_id, first_line, found_type)

            # strict integer filter
            if found_type == 'multi_level_heading' and raw_id.isdigit():
                has_dot       = first_line.strip().startswith(f"{raw_id}.")
                rest          = first_line.strip()[len(raw_id):].strip()
                is_clean_title = re.match(r'^[A-Z][A-Z\s\(\)]{2,}$', rest)
                if not has_dot and not is_clean_title:
                    # Reject immediately if the text right after the number starts with a
                    # lowercase letter — that is running prose (e.g. "95 percent eye range
                    # contour means..."), not a heading title.
                    if rest and rest[0].islower():
                        continue
                    # Also allow bare integers where the title/body appears on the next line
                    # e.g. "98\nUniform provisions concerning..." (UN Regulation numbers)
                    remaining_lines = [ln.strip() for ln in content.splitlines()[1:] if ln.strip()]
                    has_body_after = remaining_lines and (
                        len(remaining_lines[0].split()) >= 3
                        or (remaining_lines[0].isupper() and len(remaining_lines[0]) > 2)
                    )
                    if not has_body_after:
                        continue

            # FMVSS bold check
            if found_type == 'fmvss_paragraph':
                is_id_bold = False
                # Trust PaddleOCR's structural heading label as sufficient evidence,
                # because sub-clauses like "S4.1.1  Frequency." are often typeset in
                # a regular (non-bold) weight even though they are real clause headings.
                if chunk.get('_block_label') in ('doc_title', 'paragraph_title', 'title'):
                    is_id_bold = True
                elif '.' in raw_id or re.match(r'^S\d+$', raw_id):
                    # FMVSS S-clauses at any level (S1, S2, S4.1, S4.1.1, …) are often
                    # not bold — the regex match alone is sufficient evidence.
                    is_id_bold = True
                elif rich_spans:
                    for s in rich_spans[:3]:
                        if raw_id in s['text'] and s['is_bold']:
                            is_id_bold = True
                            break
                if not is_id_bold:
                    continue

            # digit-only safety checks
            is_digit_only = raw_id.isdigit()
            line_content  = self._strip_heading_prefix(first_line, raw_id)

            # Reject OCR noise lines where the "title" portion after the clause
            # number starts with digits and contains non-word characters (e.g.
            # "3 51/i" → line_content="51/i").  Real headings never look like
            # that — a leftover number fragment with slashes or similar junk is
            # a sure sign the OCR picked up a watermark, cross-reference, or
            # page-layout artifact rather than an actual clause heading.
            if (is_digit_only and line_content
                    and re.match(r'^\d', line_content)
                    and re.search(r'[^A-Za-z0-9\s\.\-]', line_content)):
                continue

            if is_digit_only and not line_content and len(raw_id) <= 3:
                # Allow "2.\n<body text>" pattern: clause number alone on first line
                # with body content on subsequent lines is a valid numbered clause.
                # Also accept a single all-caps title on the next line (e.g. "1.\nSCOPE").
                remaining_lines = [ln.strip() for ln in content.splitlines()[1:] if ln.strip()]
                has_body_after = remaining_lines and (
                    len(remaining_lines[0].split()) >= 3
                    or (remaining_lines[0].isupper() and len(remaining_lines[0]) > 2)
                    or remaining_lines[0].strip().endswith(':')
                )
                if not has_body_after:
                    continue
            if is_digit_only and len(raw_id) == 4 and (1900 <= int(raw_id) <= 2050):
                continue
            if is_digit_only and len(raw_id) <= 2 and month_pattern.search(first_line):
                continue

            # TOC
            if found_type == 'toc_start_heading':
                chunk['clause_id'] = "TOC"
                chunk['title']     = "Table of Contents"
                chunk['text_type'] = 'toc'
                chunk['level']     = 0
                continue

            detected_title = self._extract_title_from_line(rich_spans, raw_id)
            if detected_title.strip() in [':', '.', '-', '']:
                detected_title = "NIL"
            # For paren/list headings (e.g. "(1)\nApproval is..."), the number sits
            # alone on the first line and the inline bold term is on the second line.
            # Fall back to the second line's rich spans when the first gives nothing.
            if detected_title == "NIL" and found_type in (
                'numeric_paren_heading', 'alpha_paren_heading',
                'roman_lower_paren_heading', 'alpha_bare_heading',
                'roman_lower_bare_heading',
            ):
                second_spans = chunk.get('_second_line_rich_spans', [])
                if second_spans:
                    detected_title = self._extract_title_from_line(second_spans, None)
                    if detected_title.strip() in [':', '.', '-', '']:
                        detected_title = "NIL"
            # For bare digit headings (e.g. "0.\nGENERAL"), the title is on the next line
            if detected_title == "NIL" and found_type == 'multi_level_heading' and raw_id.isdigit():
                next_lines = [ln.strip() for ln in content.splitlines()[1:] if ln.strip()]
                if next_lines and next_lines[0].isupper() and len(next_lines[0]) > 2:
                    detected_title = next_lines[0]
            # For CFR clause-with-year headings (e.g. "86.135–90" alone, title on next line)
            if found_type == 'cfr_clause_year_heading':
                next_lines = [ln.strip() for ln in content.splitlines()[1:] if ln.strip()]
                detected_title = next_lines[0] if next_lines else "NIL"
            # For word-based headings the raw_id is a keyword (SCOPE, INTRODUCTION…),
            # not a number prefix — the full first line is the title, not the remainder.
            if found_type in (
                'content_start_heading', 'preamble_heading', 'chapter_heading',
                'roman_upper_heading', 'roman_upper_alpha_heading', 'appendix_heading',
            ):
                detected_title = first_line
                # When roman numeral is alone on its line (e.g. "I.\nSome Title"),
                # use the next line as the title instead of bare "I." or "Ia."
                if found_type in ('roman_upper_heading', 'roman_upper_alpha_heading') and \
                        detected_title.strip().rstrip('.') == raw_id:
                    next_lines = [ln.strip() for ln in content.splitlines()[1:] if ln.strip()]
                    if next_lines:
                        detected_title = next_lines[0]

            # level computation
            level = 1
            if found_type == 'article_heading':
                level = 0
            elif found_type in ('content_start_heading', 'preamble_heading',
                                 'chapter_heading', 'appendix_heading', 'roman_upper_heading'):
                level = 1
            elif found_type == 'roman_upper_alpha_heading':
                level = 2  # sub-item of a roman-numeral section (e.g. Ia. under I.)
            elif found_type == 'cfr_clause_year_heading':
                # Count dots in the base number only (ignore the –year suffix)
                base = re.split(r'[–\-—]', raw_id)[0]
                level = base.count('.') + 1
            elif found_type == 'multi_level_heading':
                level = raw_id.count('.') + 1
                if raw_id.isdigit() and detected_title == "NIL":
                    level = 1   # bare integers without a title stay at level 1
            elif found_type == 'fmvss_paragraph':
                # S4 → level 2, S4.1 → level 3, S4.1.1 → level 4, etc.
                # FMVSS S-clauses are children of the §-section heading (level 1).
                level = raw_id.count('.') + 2
            elif found_type in ('numeric_paren_heading', 'alpha_paren_heading',
                                 'alpha_bare_heading'):
                level = 2       # refined by _assign_parent_ids stack
            elif found_type in ('roman_lower_paren_heading', 'roman_lower_bare_heading'):
                level = 3       # roman sub-items nest one level deeper
            elif found_type == 'bullet_heading':
                level = 3       # bullets are leaf-level list items

            _tt = self._determine_text_type(
                found_type, chunk.get('_block_label', ''), first_line
            )

            chunk['clause_id']  = raw_id
            chunk['title']      = detected_title
            chunk['level']      = level
            chunk['text_type']  = _tt
            chunk['_found_type'] = found_type  # retained for downstream parent logic

        # ── Pass 1a: normalize text_type from _block_label for special types ──
        # Blocks are created with text_type='text' regardless of PaddleOCR label.
        # Update text_type for figure_title and image_caption blocks that were
        # not matched by any heading regex in Pass 1, so downstream steps
        # (_merge_by_left_alignment, _assign_parent_ids) handle them correctly.
        for chunk in raw_chunks:
            if chunk.get('clause_id') is not None:
                continue
            _lbl = (chunk.get('_block_label') or '').lower()
            if 'figure_title' in _lbl:
                chunk['text_type'] = 'figure_title'
            elif 'image_caption' in _lbl or _lbl == 'caption':
                chunk['text_type'] = 'image_caption'

        # ── Pass 1b: clause_id + Title for unmatched structural/all-caps blocks ─
        # Blocks that carry no regex clause_id but whose first line is all-caps
        # or is labelled as a structural heading by PaddleOCR are promoted:
        #   clause_id = first_line   (so they're referenceable, like SCOPE)
        #   Title     = first_line
        #   level     = 1            (parallel to top-level numbered clauses)
        # Track the _found_type of the most-recently-seen Pass-1 heading so that
        # Pass 1b can detect chapter-subtitle blocks (all-caps text immediately
        # following a chapter/roman heading that is the heading's title, not a
        # new top-level clause).
        _CHAPTER_TYPES = frozenset({
            'chapter_heading', 'roman_upper_heading',
            'roman_upper_alpha_heading', 'appendix_heading',
        })
        _prev_found_type = None
        for chunk in raw_chunks:
            # Update prev-heading tracker BEFORE the early-continue so we always
            # see the found_type of Pass-1 annotated chunks.
            _ft = chunk.get('_found_type')
            if _ft:
                _prev_found_type = _ft
            if chunk.get('clause_id') is not None:
                continue
            if chunk.get('text_type') in ('table', 'figure', 'figure_title', 'image_caption', 'toc', 'header', 'footer', 'footnote', 'vision_footnote'):
                continue
            content    = chunk.get('content_verbatim', '')
            first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), '')
            if not first_line:
                continue
            rich_spans = chunk.get('_first_line_rich_spans', [])
            is_all_caps     = first_line.isupper() and len(first_line) > 2
            is_bold_span    = any(s.get('is_bold') for s in rich_spans)
            is_paddle_heading = chunk.get('_block_label') in (
                'doc_title', 'paragraph_title', 'title'
            )
            # Skip bare numeric strings (page numbers, document IDs, regulation
            # numbers in headers). Pass 1 already handles numbered clauses like
            # "2." or "3.1"; a lone digit-only string here is always a header artefact.
            is_pure_number = re.fullmatch(r'\d+', first_line) is not None
            # Reject approval-number examples and reference codes that happen to be
            # all-uppercase but contain characters that never appear in real clause
            # identifiers (e.g. "E11*[XXX]R01/00/02*0123*01").
            if re.search(r'[*\[\]]', first_line):
                continue
            # Reject promotion when the first line is a long sentence — real headings
            # are short labels (e.g. "SCOPE", "DEFINITIONS"). A line longer than
            # 80 characters is almost certainly body text, regardless of bold.
            # Bold body text is common in regulatory PDFs (e.g. scope paragraphs).
            # Only allow long-line promotion when:
            #   (a) PaddleOCR explicitly labels the block as a structural heading, OR
            #   (b) the first line is entirely upper-case (e.g. doc titles like
            #       "UNIFORM PROVISIONS CONCERNING THE APPROVAL OF...").
            # Bold-only is deliberately NOT enough for long sentences.
            is_long_sentence = len(first_line) > 80
            if is_long_sentence and not is_paddle_heading and not is_all_caps:
                continue
            # Reject noise fragments: status markers, punctuation, and very short
            # abbreviations ("pp", "y", ",") that appear as separate OCR blocks in
            # amendment lists or footnotes.  A real heading must either:
            #   (a) contain at least one word of 3+ letters, OR
            #   (b) look like a clause number ("1.", "2.3.", "3.1.2") — these have
            #       no letters but are valid numbered clause identifiers.
            # Single chars, 2-char codes, and bare punctuation are rejected.
            is_clause_number_frag = bool(re.fullmatch(r'\d[\d.]*\.?', first_line.strip()))
            has_real_word = bool(re.search(r'[A-Za-z]{3,}', first_line)) or is_clause_number_frag
            if not has_real_word:
                continue
            # For bold-only promotions (not paddle-labeled headings, not all-caps),
            # also reject lead-in sentences like "Incorporating all valid text up to:"
            # which are body text introducing a list.  Real bold headings are short
            # identifiers (e.g. "General", "Note:", "Type Approval"); cap at 5 words
            # for bold-only so multi-word sentences are never mistaken for headings.
            if is_bold_span and not is_paddle_heading and not is_all_caps:
                word_count = len(first_line.split())
                if word_count > 5:
                    continue
            # An all-caps block immediately following a chapter/roman/appendix heading
            # is the chapter's own title (e.g. "CHAPTER II\nTYPE-APPROVAL IN ACCORDANCE
            # WITH ESSENTIAL REQUIREMENTS BASED ON UN REGULATIONS").  Promoting it as
            # a new level-1 clause would incorrectly pop the chapter off the parent
            # stack in _assign_parent_ids.  Leave clause_id=None so it inherits its
            # parent from the chapter heading already on the stack.
            if is_all_caps and _prev_found_type in _CHAPTER_TYPES and not is_paddle_heading:
                continue
            # Reject figure/table label lines like "Figure B94–1" or "Table 3.2".
            # These are captions or internal figure titles, not document headings.
            if re.match(r'(?i)^(figure|table|fig\.?)\s+[A-Z0-9][A-Z0-9\-–—\.]*\s*$', first_line):
                continue
            # Short lines ending in ":" are section labels (e.g. "Attachments:").
            # Promote them so they get their own chunk and act as parents for
            # any immediately following indented content.
            is_label_heading = (
                first_line.endswith(':') and
                len(first_line.split()) <= 4 and
                len(first_line) <= 60 and
                not is_pure_number
            )
            if (is_all_caps or is_bold_span or is_paddle_heading or is_label_heading) and not is_pure_number:
                chunk['clause_id'] = first_line
                chunk['title']     = first_line
                chunk['level']     = 1
                chunk['text_type'] = 'heading'
                if is_paddle_heading and not chunk.get('_found_type'):
                    chunk['_found_type'] = 'doc_title_heading'

        # ── Pass 2: TOC propagation ───────────────────────────────────────────
        in_toc            = False
        toc_done          = False   # once we exit TOC mode, never re-enter
        toc_ids_seen      = set()
        toc_entry_titles  = {}      # maps normalized id → first word of its TOC title
        toc_anchor_pages  = set()   # pages covered by the active TOC anchor

        _annex_pat = re.compile(r'(?i)^(annex|appendix|appendices|annexure)\b')

        # Matches known document-body section titles (case-insensitive).
        # Used by Rule 3 to detect when a heading is a real section start,
        # not merely a TOC listing of that section's title.
        _content_start_pat = re.compile(
            r'(?i)^\s*'
            r'(?:[IVXLCDM]{1,8}\.?\s+)?'          # optional roman-numeral prefix
            r'(?:executive\s+)?'                    # optional "Executive" qualifier
            r'(introduction|foreword|preamble|purpose|scope|definitions?|'
            r'general\s+provisions?|background|overview|summary|preface|'
            r'references?|normative\s+references?|terms\s+and\s+definitions?)\b'
        )

        def _has_body_prose(content):
            """Return True if the chunk contains at least one line of body prose.

            Body prose is text that is NOT:
              - a sub-clause reference  (e.g. "C1.", "Appendix 1.", "1.", "(i)")
              - a dotted TOC leader     (e.g. "............ 12")
              - a trailing page number  (e.g. "  12")
              - a short parenthetical   (e.g. "(Reserved)")
            AND has 6 or more words.

            Only lines after the first (heading) line are examined so that a
            heading like "3. Introduction" alone does not falsely trigger.
            """
            _ref_line = re.compile(
                r'(?i)^(?:'
                r'\d{1,3}(?:\.\d+)*\.?\s+'            # "1.", "3.2."
                r'|[A-Za-z]\d+\.?\s+'                  # "C1.", "A2."
                r'|\([^)]{0,30}\)\s*$'                 # "(Reserved)", "(i)"
                r'|(?:appendix|appendices|annex|annexes|annexure|'
                r'chapter|section|part|table|figure)\b'  # structural keyword starts
                r')'
            )
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            for ln in lines[1:]:               # skip the heading line itself
                if re.search(r'(\.{3,}|\s\d+\s*$)', ln):
                    continue                   # dotted leader or trailing page number
                if _ref_line.match(ln):
                    continue                   # looks like a sub-reference listing
                if len(ln.split()) >= 6:
                    return True                # substantial prose → body content
            return False

        # A TOC entry line must start with a structural identifier followed by
        # a CAPITAL letter.  Requiring the capital prevents body-prose lines like
        # "111 to include a 10-foot zone" from being mistaken for section numbers.
        _TOC_ENTRY_RE = re.compile(
            r'^(?:'
            r'[IVXLCDM]{1,8}\.?\s+[A-Z]'        # Roman: I. Executive  II. Background
            r'|[a-z]\.\s+[A-Z]'                   # Letter: a. Cameron   b. Safety
            r'|\d{1,3}(?:\.\d+)*\.?\s+[A-Z]'     # Number: 1. General   2.1 Scope
            r')'
        )

        def _split_toc_from_prose(content_verbatim):
            """Split a mixed TOC+prose block into (toc_lines, prose_lines).

            The block must START with a structural TOC identifier followed by a
            capital letter (see _TOC_ENTRY_RE).  If the first line does not
            match, the whole content is treated as prose and ('', content_verbatim)
            is returned — this prevents body paragraphs that wrap into short lines
            (e.g. "111 to include a 10-foot zone") from being misidentified.

            Once a valid TOC start is confirmed, scans forward: lines that start
            with a structural identifier (capital required) or are very short
            continuations (≤ 2 words, e.g. "Event") are kept as TOC lines.
            Two consecutive substantive non-structural lines (≥ 3 words) mark
            the prose boundary.
            """
            lines = [ln for ln in content_verbatim.splitlines() if ln.strip()]
            if not lines or not _TOC_ENTRY_RE.match(lines[0].strip()):
                return '', content_verbatim   # first line not structural → pure prose
            split_idx = len(lines)
            consecutive_non_structural = 0
            for i, ln in enumerate(lines):
                stripped = ln.strip()
                if _TOC_ENTRY_RE.match(stripped) or len(stripped.split()) <= 2:
                    consecutive_non_structural = 0
                else:
                    consecutive_non_structural += 1
                    if consecutive_non_structural >= 2:
                        split_idx = i - (consecutive_non_structural - 1)
                        break
            return '\n'.join(lines[:split_idx]), '\n'.join(lines[split_idx:])

        def _bbox_min_y(block_bboxes):
            """Return the minimum y0 across all bboxes in a block_bboxes dict."""
            min_y = float('inf')
            for pg_boxes in (block_bboxes or {}).values():
                for bb in (pg_boxes or []):
                    if len(bb) >= 4:
                        min_y = min(min_y, bb[1])
            return min_y if min_y != float('inf') else 0

        def _toc_first_word(content):
            """Return the lowercase first word of the title in a TOC/heading entry.

            Skips any leading bare-number lines (e.g. "1." or "11") and strips a
            leading number prefix from the title line so that both "1\\nScope" and
            "1.\\nScope" return "scope".
            """
            for ln in (l.strip() for l in content.splitlines() if l.strip()):
                if re.match(r'^[\d.]+$', ln):
                    continue                         # pure number — skip
                m2 = re.match(r'^[\d.]+\s+(.*)', ln)
                title_raw = (m2.group(1) if m2 else ln).strip().lower()
                words = title_raw.split()
                return words[0] if words else ''
            return ''

        def _next_same_page_chunk_has_body_prose(start_idx, page_keys):
            """
            Return True when the next substantive chunk on the same page(s)
            clearly begins document-body prose.

            This handles PDFs where a real section heading such as
            "INTRODUCTION" is split into its own block and the paragraph text
            starts in the immediately following chunk on the same page.
            """
            if not page_keys:
                return False

            for next_chunk in raw_chunks[start_idx + 1:]:
                next_pages = set(next_chunk.get('block_bboxes', {}).keys())
                if next_pages and not next_pages.intersection(page_keys):
                    break

                if next_chunk.get('text_type') in (
                    'header', 'footer', 'footnote', 'vision_footnote'
                ):
                    continue

                next_text = next_chunk.get('content_verbatim', '').strip()
                if not next_text:
                    continue

                # Another explicit TOC anchor means we should not infer body prose
                # from later chunks.
                if next_chunk.get('clause_id') == "TOC":
                    return False

                if next_chunk.get('text_type') in ('table', 'figure', 'toc'):
                    continue

                if _has_body_prose(next_text):
                    return True

                # Stop at the next substantive non-prose structural block rather
                # than scanning arbitrarily far down the page.
                if (next_chunk.get('clause_id') is not None
                        or next_chunk.get('_found_type')
                        or next_chunk.get('text_type') == 'heading'):
                    return False

            return False

        for idx, chunk in enumerate(raw_chunks):
            if chunk.get('clause_id') == "TOC":
                # Re-activate TOC mode for every explicit TOC anchor.
                # Documents with embedded Appendices (e.g. Appendix A) may
                # contain their own TOC in the middle of the document; each
                # such anchor must restart propagation with a clean state.
                in_toc           = True
                toc_done         = False
                toc_ids_seen     = set()
                toc_entry_titles = {}
                toc_anchor_pages = set(chunk.get('block_bboxes', {}).keys())
                continue

            if not in_toc:
                continue

            # ── Skip structural page-edge elements inside the TOC ─────────────
            # Running headers, footers, and footnotes can appear between TOC
            # entries (e.g. a page header on the second TOC page).  They are
            # never TOC content, so we skip them entirely — without touching
            # toc_ids_seen, toc_entry_titles, or any exit-condition logic.
            # This lets TOC propagation continue uninterrupted to the next
            # genuine content chunk.
            if chunk.get('text_type') in ('header', 'footer', 'footnote', 'vision_footnote'):
                continue

            cid = chunk.get('clause_id')
            # Treat sentinel "NIL" as absent — it must not pollute toc_ids_seen
            # or trigger the duplicate-detection exit from TOC mode.
            if isinstance(cid, str) and cid.upper() == "NIL":
                cid = None
            _cv_stripped = chunk['content_verbatim'].strip()
            is_visual_toc = bool(
                re.search(r'(\.{3,}|\s\d+)$', _cv_stripped)
                or any(re.search(r'\.{10,}', ln)
                       for ln in _cv_stripped.splitlines() if ln.strip())
            )

            if cid is not None:
                cid_clean = cid.lower().replace(" ", "")
                # Normalise: strip trailing dot so "1." and "1" compare equal.
                cid_norm  = cid_clean.rstrip('.')

                # Current chunk's title first-word (used for duplicate disambiguation)
                curr_first_word = _toc_first_word(chunk.get('content_verbatim', ''))

                # Exact match OR prefix match (e.g. "REGULATION" in toc_ids_seen
                # should also catch body heading "REGULATION No. 14-08").
                # Guard: don't let "annex1" match "annex10" — if the next char
                # after the seen prefix is a digit, it's a different numbered item.
                # Also guard against compound-word false matches: "annexesparta"
                # starts with "annexes" but the next char "p" is alphabetic, meaning
                # it's a longer compound id, not the same clause (e.g. "Annexes Part A"
                # must not be treated as a duplicate of the "Annexes" heading).
                exact_dup  = cid_norm in toc_ids_seen
                prefix_dup = any(
                    cid_norm.startswith(seen)
                    and not cid_norm[len(seen):len(seen) + 1].isdigit()
                    and not cid_norm[len(seen):len(seen) + 1].isalpha()
                    for seen in toc_ids_seen if len(seen) >= 4
                )
                is_dup = exact_dup or prefix_dup

                if is_dup and not is_visual_toc:
                    # For exact-id duplicates, only exit when the chunk's title
                    # matches the title we saw for that id the FIRST time.  A
                    # mismatch means the TOC contains a sub-list that reuses the
                    # same clause numbers (e.g. Annex listings numbered 1–11
                    # after a main section list also numbered 1–13).  In that
                    # case we must NOT exit — just continue tagging as TOC.
                    #
                    # For prefix matches (long keyword ids) fall through to the
                    # original exit behaviour since there is no clean title to compare.
                    should_exit = True
                    # Structural section labels (Appendix, Annex, Annexes…) repeat
                    # naturally across TOC sub-lists — a complex TOC may have many
                    # "Appendix" sub-headings (one per Annex Part).  Never exit on
                    # these regardless of title match.
                    _structural_toc_labels = {
                        'appendix', 'appendices', 'annex', 'annexes', 'annexure'
                    }
                    if cid_norm in _structural_toc_labels:
                        should_exit = False
                    elif exact_dup and not prefix_dup:
                        stored_fw = toc_entry_titles.get(cid_norm, '')
                        if stored_fw and curr_first_word and curr_first_word != stored_fw:
                            should_exit = False  # title mismatch → still in a TOC sub-list
                    if should_exit:
                        in_toc = False; toc_done = True   # real clause starts here
                        continue

                # Rule 3: Exit when a heading matches a known document-body section
                # title (e.g. "Introduction", "SCOPE", "Definitions") AND the chunk
                # contains at least one line of body prose beneath it.
                #
                # A TOC *listing* of "Introduction" looks like:
                #   "Introduction .......... 3"
                # and will be caught by is_visual_toc before reaching here.
                # A real body section looks like:
                #   "Introduction\nThis regulation applies to all vehicles…"
                # and has prose that _has_body_prose() detects.
                #
                # Annex/Appendix headings are deliberately NOT excluded here —
                # the Annex listing inside a TOC ("Annexes Part C\nC1. (Reserved)")
                # never has body prose, so _has_body_prose() keeps us inside the TOC.
                _chunk_pages_cid = set(chunk.get('block_bboxes', {}).keys())
                _on_toc_page_cid = bool(
                    _chunk_pages_cid and _chunk_pages_cid.issubset(toc_anchor_pages)
                )
                if not is_visual_toc and _content_start_pat.match(cid):
                    _next_same_page_body = _next_same_page_chunk_has_body_prose(
                        idx, _chunk_pages_cid
                    )
                    if _has_body_prose(chunk.get('content_verbatim', '')):
                        in_toc = False; toc_done = True   # real section body starts here
                        continue
                    elif _next_same_page_body:
                        in_toc = False; toc_done = True   # heading-only block; prose begins in next chunk
                        continue
                    elif not _on_toc_page_cid:
                        # Off the TOC page → real section heading (prose in next block).
                        # Grow pages so later entries on this page are still collected.
                        toc_anchor_pages.update(_chunk_pages_cid)
                        continue
                    else:
                        # Still on the TOC page → keyword title is a TOC listing,
                        # not the real section start.  Tag it as toc so Pass 2b
                        # absorbs it into the TOC anchor.
                        pass  # fall through to toc tagging below

                # A TOC always lists sections in ascending order.  If the current
                # chunk's clause_id is a bare integer LOWER than the smallest integer
                # already recorded in toc_ids_seen (e.g. we've seen "2","3",..."7"
                # but "1" never appeared because its TOC entry had no number prefix),
                # this chunk cannot be a TOC entry — it must be body content.
                # Visual-TOC entries (dotted leaders / trailing page numbers) are
                # excluded via is_visual_toc — genuine late TOC sub-entries always
                # have dots and a page number, so they won't falsely trigger this.
                if not is_dup and not is_visual_toc:
                    try:
                        cid_int = int(cid_norm)
                        seen_ints = {int(s) for s in toc_ids_seen if re.fullmatch(r'\d+', s)}
                        if seen_ints and cid_int < min(seen_ints):
                            in_toc = False; toc_done = True   # lower-numbered section → body started earlier
                            continue
                    except ValueError:
                        pass

                # Record first-occurrence title (don't overwrite with later sub-list entry)
                if cid_norm not in toc_entry_titles:
                    toc_entry_titles[cid_norm] = curr_first_word
                toc_ids_seen.add(cid_norm)
                chunk['clause_id'] = None
                chunk['text_type'] = 'toc'
                # Grow toc_anchor_pages to include this chunk's page(s) so that
                # later entries on the same page are also seen as "on TOC page".
                toc_anchor_pages.update(set(chunk.get('block_bboxes', {}).keys()))
            else:
                # cid is None — two sub-cases reaching here
                # (header/footer/footnote were already skipped above):
                # (A) Legitimate TOC content: a split TOC entry or un-annotated
                #     sub-listing line.  Tag as toc.
                # (B) Real body heading mislabelled by PaddleOCR as a plain 'text'
                #     block and therefore skipped in Pass 1 (e.g. "1.\nLEGISLATIVE
                #     PROVISIONS" without dotted leaders).  This should EXIT toc mode.
                #
                # Distinguish (B) by: number-only first line, non-empty second line
                # that contains no dotted leaders/trailing page number, AND the number
                # was already collected from a real TOC entry above.
                _cv_lines = [ln.strip()
                             for ln in chunk.get('content_verbatim', '').splitlines()
                             if ln.strip()]
                _fl         = _cv_lines[0] if _cv_lines else ''
                _num_m      = re.match(r'^(\d{1,3}(?:\.\d+)*)\.?\s*$', _fl)
                _alpha_num_m = re.match(r'^([A-Za-z]\d+[a-z]?)\.?\s*$', _fl)
                _clause_m   = _num_m or _alpha_num_m   # catches "1.", "3.2.", "A1.", "B6a."
                _title      = _cv_lines[1] if len(_cv_lines) > 1 else ''
                # Guard against appendix/sub-list TOC entries such as:
                #   "2\nCommunication\nAppendix 1 - Reserved\nAppendix 2 - ..."
                # These often miss clause_id annotation, but they are still TOC
                # content and must not terminate TOC mode. A real body heading
                # should be compact here: usually just the number and title line,
                # optionally followed later by separate body-text chunks.
                _tail_lines = _cv_lines[2:] if len(_cv_lines) > 2 else []
                _has_toc_sublist_tail = any(
                    re.match(
                        r'(?i)^(appendix|appendices|annex|annexes|annexure|'
                        r'chapter|section|part|table|figure)\b',
                        ln,
                    )
                    for ln in _tail_lines
                )
                # A cid=None chunk that contains body prose (not just sub-reference
                # lines) is document body content, not a TOC entry.  Exit
                # regardless of whether annex entries have been seen — complex
                # Annex sub-listings ("Annexes Part C\nC1. (Reserved)\n...") never
                # contain prose and will safely stay inside the TOC.
                #
                # Two guards suppress premature exit:
                #   1. _alpha_num_m: first line is an annex sub-clause ref like
                #      "A1.", "B6a.", "C1." — its title may be a long descriptive
                #      phrase that looks like prose but is still a TOC entry.
                #   2. on_toc_page: the chunk sits on the same page(s) as the TOC
                #      anchor, so even if _has_body_prose fires, we stay in mode.
                _chunk_pages  = set(chunk.get('block_bboxes', {}).keys())
                _on_toc_page  = bool(toc_anchor_pages and _chunk_pages.issubset(toc_anchor_pages))

                # Rule 3b: Exit when a cid=None chunk's first line is a known
                # body-section title (Introduction, Scope, Foreword, Preamble,
                # etc.) without any visual TOC markers (dotted leaders /
                # trailing page number).  Do NOT require body prose in the
                # same block — PaddleOCR often splits the title heading into
                # its own block separate from the prose paragraphs, so
                # _has_body_prose would return False on the heading-only block.
                # However, if the chunk is on the TOC page itself, a standalone
                # keyword title (e.g. "Introduction", "Scope") is almost certainly
                # a TOC entry — not the real section start — so only exit when we
                # are off the TOC pages OR the block also contains body prose.
                if (not is_visual_toc and _content_start_pat.match(_fl)
                        and (not _on_toc_page
                             or _next_same_page_chunk_has_body_prose(idx, _chunk_pages)
                             or _has_body_prose('\n'.join(_cv_lines)))):
                    in_toc = False; toc_done = True
                    continue

                # A prose block with >= 4 lines is unmistakably body content and
                # overrides the _on_toc_page locality guard.  TOC sub-entries are
                # always short (1-2 lines per entry); multi-paragraph body text
                # cannot be a TOC entry regardless of which page it sits on.
                _long_prose_block = len(_cv_lines) >= 4
                if (not is_visual_toc
                        and not _alpha_num_m
                        and (not _on_toc_page or _long_prose_block)
                        and chunk.get('text_type') not in (
                            'header', 'footer', 'footnote', 'table', 'figure', 'vision_footnote')
                        and _has_body_prose('\n'.join(_cv_lines))):
                    in_toc = False; toc_done = True
                    continue

                if (_clause_m and _title
                        and not _has_toc_sublist_tail
                        and not re.search(r'(\.{3,}|\s\d+\s*$)', _title)
                        and _clause_m.group(1).lower() in toc_ids_seen):
                    # Mirror the duplicate-id disambiguation used for cid-bearing
                    # TOC items above: only exit when this clause heading's title
                    # matches the first title we saw for that same id.
                    # This keeps appendix/annex sub-lists like "7\nReserved" and
                    # "A1\nEngine..." inside the TOC when annotation failed.
                    _clause_id = _clause_m.group(1).lower()
                    _stored_fw = toc_entry_titles.get(_clause_id, '')
                    _curr_fw = _title.lower().split()[0] if _title.strip() else ''
                    if _stored_fw and _curr_fw and _curr_fw == _stored_fw:
                        # Real body section heading — exit TOC mode without tagging
                        in_toc = False; toc_done = True
                        continue

                # Tag as TOC content (headers/footers/footnotes were already
                # skipped at the top of the loop; table/figure remain as-is).
                if chunk.get('text_type') not in ('table', 'figure'):
                    chunk['text_type'] = 'toc'
                    # Grow toc_anchor_pages to include this chunk's page(s) so
                    # that subsequent entries on the same page are also seen as
                    # "on TOC page" and their body-prose exit is suppressed.
                    toc_anchor_pages.update(_chunk_pages)
                first_line = _fl
                if _clause_m:
                    id_str = _clause_m.group(1).lower().replace(' ', '')
                    toc_ids_seen.add(id_str)
                    # Record first-occurrence title for this id
                    if id_str not in toc_entry_titles and _title:
                        toc_entry_titles[id_str] = _title.lower().split()[0]

        # ── Pass 2b: Merge toc-tagged chunks into their nearest TOC anchor ──
        # After Pass 2 marks individual blocks as text_type='toc', we consolidate
        # them into the TOC anchor that immediately precedes them.  There may be
        # more than one TOC anchor in the document (e.g. the main TOC plus a
        # separate TOC embedded in Appendix A), so we process each anchor
        # independently and assign each toc-tagged chunk to its nearest preceding
        # anchor.
        toc_anchor_indices = [i for i, c in enumerate(raw_chunks)
                              if c.get('clause_id') == "TOC"]

        if toc_anchor_indices:
            absorbed = set()

            for anchor_idx in toc_anchor_indices:
                anchor    = raw_chunks[anchor_idx]
                toc_pages = set(anchor.get('block_bboxes', {}).keys())

                # Block-level reading-order tracker.  Each entry is (y_min, text).
                # We sort by y_min at the end so absorbed chunks land at the right
                # position regardless of the order they are encountered here.
                _toc_blocks = [
                    (_bbox_min_y(anchor.get('block_bboxes', {})),
                     anchor.get('content_verbatim', '').strip())
                ]

                # Absorb everything between this anchor and the next one.
                next_anchor = next(
                    (nai for nai in toc_anchor_indices if nai > anchor_idx),
                    len(raw_chunks),
                )

                for idx in range(anchor_idx + 1, next_anchor):
                    if idx in absorbed:
                        continue
                    chunk = raw_chunks[idx]
                    is_toc_tagged = (
                        chunk.get('text_type') == 'toc'
                        and chunk.get('clause_id') != "TOC"
                    )
                    # Also absorb chunks that fall entirely within the TOC's page
                    # range and whose clause_id looks like an Annex/Appendix entry.
                    # These are entries (headings OR text blocks) that appear on the
                    # TOC pages but were not tagged as toc by Pass 2 — e.g. Annex 7,
                    # Annex 8, "Appendix 1: Location..." which are sub-listed items
                    # on the TOC page.  We intentionally do NOT restrict by text_type
                    # because PaddleOCR sometimes detects these as 'text' blocks.
                    chunk_pages = set(chunk.get('block_bboxes', {}).keys())
                    cid_str     = str(chunk.get('clause_id') or '')
                    is_annex_on_toc_pages = (
                        bool(chunk_pages)
                        and chunk_pages.issubset(toc_pages)
                        and chunk.get('text_type') not in ('table', 'figure', 'header', 'footer', 'footnote', 'vision_footnote')
                        and bool(re.search(r'(?i)^(annexes?|appendix|appendices|annexure)\b', cid_str))
                    )
                    # Catch any remaining TOC-page entries that weren't tagged by
                    # Pass 2 — e.g. "Scope" or "Definitions" whose keyword titles
                    # caused Pass 2 to treat them as real section headings and skip
                    # toc-tagging entirely.  Rule: if the chunk sits wholly on the
                    # TOC page(s), has no body prose, and is not a structural element,
                    # absorb it unconditionally.
                    is_plain_toc_page_entry = (
                        bool(chunk_pages)
                        and chunk_pages.issubset(toc_pages)
                        and chunk.get('clause_id') not in ('TOC', 'PRELUDE')
                        and chunk.get('text_type') not in (
                            'table', 'figure', 'header', 'footer',
                            'footnote', 'vision_footnote', 'prelude'
                        )
                        and not _has_body_prose(chunk.get('content_verbatim', ''))
                    )
                    # Guard: never absorb a chunk whose content contains a
                    # known body-section title (Introduction, Scope, Foreword,
                    # etc.) followed by body prose.  This catches two cases:
                    #   (A) The Introduction is in its own chunk but was
                    #       mis-tagged as toc by Pass 2 (e.g. page-locality
                    #       suppressed the exit).
                    #   (B) PaddleOCR merged the last TOC entry and the
                    #       Introduction into the same OCR block; is_toc_tagged
                    #       would be True but absorbing would bury the prose.
                    # We also handle the split-content case: if the chunk
                    # contains both TOC lines and Introduction+prose, strip the
                    # body portion before absorbing the TOC lines.
                    #
                    # _whole_chunk_is_body: set True when we have determined the
                    # chunk (or its rewritten content) is entirely body prose.
                    # Blocks the is_mixed_toc_page_entry path below from running
                    # _split_toc_from_prose again and wrongly pulling a heading
                    # line (like "I. Executive Summary") into the TOC.
                    _whole_chunk_is_body = False
                    if is_toc_tagged:
                        _p3_lines = [
                            ln.strip()
                            for ln in chunk.get('content_verbatim', '').splitlines()
                            if ln.strip()
                        ]
                        _split_at = None
                        _p3_cv = chunk.get('content_verbatim', '').strip()
                        _p3_is_visual = bool(
                            re.search(r'(\.{3,}|\s\d+)$', _p3_cv)
                            or any(re.search(r'\.{10,}', ln)
                                   for ln in _p3_cv.splitlines() if ln.strip())
                        )
                        if not _p3_is_visual:
                            for _p3_i, _p3_ln in enumerate(_p3_lines):
                                # Skip lines that contain TOC leader dots — they
                                # are TOC entries (e.g. "Scope.......1"), not
                                # real body-section headings.
                                if re.search(r'\.{5,}', _p3_ln):
                                    continue
                                # A line matching a known body-section title
                                # without visual TOC markers (dots / page num)
                                # signals the start of real document content.
                                # No body-prose requirement: PaddleOCR often
                                # emits the heading as its own block, separate
                                # from the prose paragraphs.
                                if _content_start_pat.match(_p3_ln):
                                    _split_at = _p3_i
                                    break

                        if _split_at is not None:
                            if _split_at == 0:
                                # Only block absorption when the chunk is off the
                                # TOC pages or it contains real body prose.
                                # When it's still on the TOC page (e.g. a standalone
                                # "Introduction" or "Scope" TOC entry in a clean-
                                # layout TOC without visual leaders), the keyword
                                # title is almost certainly a TOC listing, not the
                                # real section start, so we keep it.
                                _p3_chunk_pages = set(chunk.get('block_bboxes', {}).keys())
                                _p3_on_toc_page = bool(
                                    _p3_chunk_pages and _p3_chunk_pages.issubset(toc_pages)
                                )
                                if not _p3_on_toc_page or _has_body_prose('\n'.join(_p3_lines)):
                                    is_toc_tagged = False
                                    _whole_chunk_is_body = True
                            else:
                                # Partial split: the chunk has TOC entries before
                                # _split_at and body prose starting at _split_at.
                                # We must absorb the TOC prefix into the anchor NOW
                                # (before is_toc_tagged is cleared) and keep the
                                # prose in the chunk so downstream passes see it.
                                # The old approach just stripped prose from the
                                # chunk without saving it, which caused data loss.
                                _tail_after_heading = _p3_lines[_split_at:]
                                _tail_has_body_prose = _has_body_prose('\n'.join(_tail_after_heading))
                                if _tail_has_body_prose and len(_tail_after_heading) >= 2:
                                    _ps_toc_cv = '\n'.join(_p3_lines[:_split_at])
                                    if _ps_toc_cv:
                                        _ps_y = _bbox_min_y(chunk.get('block_bboxes', {}))
                                        _toc_blocks.append((_ps_y, _ps_toc_cv))
                                        anchor.setdefault('content_words', []).extend(
                                            chunk.get('content_words', [])
                                        )
                                        for _ps_pg, _ps_bbs in chunk.get('block_bboxes', {}).items():
                                            _ps_pg_list = anchor.setdefault(
                                                'block_bboxes', {}
                                            ).setdefault(_ps_pg, [])
                                            for _ps_bb in _ps_bbs:
                                                if _ps_bb not in _ps_pg_list:
                                                    _ps_pg_list.append(_ps_bb)
                                    # Keep the prose in the chunk; block further
                                    # TOC absorption paths for this chunk.
                                    chunk['content_verbatim'] = '\n'.join(_tail_after_heading)
                                    is_toc_tagged = False
                                    _whole_chunk_is_body = True

                    # Fallback inside is_toc_tagged: _content_start_pat only
                    # recognises keywords like "Introduction" / "Scope" / etc.
                    # It misses bodies that start with a roman-numeral heading
                    # ("I. Executive Summary") or a name ("The Cameron Gulbransen
                    # Kids").  If the chunk is still tagged as toc-content but
                    # contains body prose, use _split_toc_from_prose to absorb
                    # only the TOC prefix and keep the prose as a separate body
                    # chunk in raw_chunks.
                    if (is_toc_tagged and _split_at is None
                            and _has_body_prose(chunk.get('content_verbatim', ''))):
                        _fb_toc, _fb_prose = _split_toc_from_prose(
                            chunk.get('content_verbatim', '')
                        )
                        if _fb_toc and _fb_prose:
                            # Add the TOC portion to the anchor now (before
                            # is_toc_tagged is cleared so the block below is skipped).
                            _fb_y = _bbox_min_y(chunk.get('block_bboxes', {}))
                            _toc_blocks.append((_fb_y, _fb_toc.strip()))
                            anchor.setdefault('content_words', []).extend(
                                chunk.get('content_words', [])
                            )
                            for _fb_pg, _fb_bbs in chunk.get('block_bboxes', {}).items():
                                _fb_pg_list = anchor.setdefault(
                                    'block_bboxes', {}
                                ).setdefault(_fb_pg, [])
                                for _fb_bb in _fb_bbs:
                                    if _fb_bb not in _fb_pg_list:
                                        _fb_pg_list.append(_fb_bb)
                            # Rewrite the chunk so downstream passes only see prose.
                            chunk['content_verbatim'] = _fb_prose
                            is_toc_tagged = False      # skip double-absorption below
                            _whole_chunk_is_body = True  # block is_mixed_toc_page_entry

                    # Fourth path: chunk is on the TOC page and STARTS with TOC
                    # entries but has body prose appended (e.g. PaddleOCR merged a
                    # section III TOC block with the start of the Executive Summary).
                    # Strip the prose portion and absorb only the TOC prefix.
                    is_mixed_toc_page_entry = False
                    _mixed_toc_cv = ''
                    if (not is_toc_tagged and not is_annex_on_toc_pages
                            and not is_plain_toc_page_entry
                            and not _whole_chunk_is_body
                            and bool(chunk_pages)
                            and chunk_pages.issubset(toc_pages)
                            and chunk.get('clause_id') not in ('TOC', 'PRELUDE')
                            and chunk.get('text_type') not in (
                                'table', 'figure', 'header', 'footer',
                                'footnote', 'vision_footnote', 'prelude'
                            )
                            and _has_body_prose(chunk.get('content_verbatim', ''))):
                        _toc_part, _prose_part = _split_toc_from_prose(
                            chunk.get('content_verbatim', '')
                        )
                        if _toc_part and _prose_part:
                            # Has a clear TOC prefix followed by prose — absorb
                            # the TOC portion; leave prose in place by rewriting
                            # the chunk so subsequent passes see only the prose.
                            _mixed_toc_cv = _toc_part
                            chunk['content_verbatim'] = _prose_part
                            is_mixed_toc_page_entry = True

                    if is_toc_tagged or is_annex_on_toc_pages or is_plain_toc_page_entry or is_mixed_toc_page_entry:
                        cv = (
                            _mixed_toc_cv if is_mixed_toc_page_entry
                            else chunk.get('content_verbatim', '')
                        ).strip()
                        if cv:
                            # Record as (y_min, text) so we can sort all absorbed
                            # blocks into correct reading order after the loop.
                            _chunk_y = _bbox_min_y(chunk.get('block_bboxes', {}))
                            _toc_blocks.append((_chunk_y, cv))
                        anchor.setdefault('content_words', []).extend(chunk.get('content_words', []))
                        for pg, boxes in chunk.get('block_bboxes', {}).items():
                            pg_list = anchor.setdefault('block_bboxes', {}).setdefault(pg, [])
                            for bb in boxes:
                                if bb not in pg_list:
                                    pg_list.append(bb)
                        anchor.setdefault('images', []).extend(chunk.get('images', []))
                        # For mixed entries the chunk is NOT absorbed — it stays in
                        # the chunk list with its prose content so downstream passes
                        # process it normally.
                        if not is_mixed_toc_page_entry:
                            absorbed.add(idx)

                # Sort collected blocks by y-position and rebuild content_verbatim.
                # This fixes the reading-order problem caused by absorption appending
                # chunks at the end regardless of where they sit on the page.
                _toc_blocks.sort(key=lambda t: t[0])
                rebuilt_cv = '\n'.join(text for _, text in _toc_blocks if text)
                if rebuilt_cv:
                    anchor['content_verbatim'] = rebuilt_cv

            raw_chunks = [c for i, c in enumerate(raw_chunks) if i not in absorbed]

        return raw_chunks

    # ─────────────────────────────────────────────────────────────────────────

    def _annotate_prelude(self, raw_chunks, prelude_page_cap=3):
        """
        STEP 2.5: Tag chunks in the document prelude with text_type='prelude'.

        The prelude is defined as everything that appears before the document's
        actual structured content begins — cover pages, title pages, copyright
        notices, document metadata, etc.

        Boundary detection (first match wins):
          WITH TOC  → everything before the TOC anchor chunk is prelude.
          NO TOC    → everything before the first level-1 structured heading
                      (a chunk with clause_id set and level >= 1), capped to
                      the first `prelude_page_cap` pages.

        Within the prelude zone, all chunks (including any typed as 'heading'
        or 'subheading') are converted to text_type='prelude'.
        """
        SKIP_TYPES = {'header', 'footer', 'footnote', 'toc', 'vision_footnote'}

        def _first_page(chunk):
            """Lowest integer page number for a chunk, or 9999 if unknown."""
            nums = []
            for p in chunk.get('block_bboxes', {}).keys():
                try:
                    nums.append(int(p))
                except (ValueError, TypeError):
                    pass
            return min(nums) if nums else 9999

        # ── WITH TOC: prelude = everything before the TOC anchor ───────────
        toc_anchor_idx = next(
            (i for i, c in enumerate(raw_chunks) if c.get('clause_id') == 'TOC'),
            None,
        )

        if toc_anchor_idx is not None:
            for chunk in raw_chunks[:toc_anchor_idx]:
                if chunk.get('text_type') in SKIP_TYPES:
                    continue
                chunk['text_type'] = 'prelude'
                chunk['clause_id'] = None   # prelude chunks carry no clause identity
                chunk['level']     = None
            return raw_chunks

        # ── NO TOC: find first structured level-1 heading as boundary ──────
        # A chunk is "structured content" if it was matched by regex (has a
        # clause_id set by Pass 1 or promoted in Pass 1b) at level >= 1.
        boundary_idx = None
        for idx, chunk in enumerate(raw_chunks):
            if chunk.get('text_type') in SKIP_TYPES:
                continue
            if _first_page(chunk) > prelude_page_cap:
                break
            # Only treat Pass 1 regex-matched headings as the real content boundary,
            # AND only when the heading is the TRUE start of the document sequence
            # (Article 1, clause 1, 1.1, etc.).  Mid-document article references
            # (e.g. "Article 28") that appear on cover pages must not fire the
            # boundary early and cut the prelude zone short.
            # CFR documents (e.g. 49 CFR Part 579) start at "§ 579.1" — the first
            # cfr_section_heading or cfr_part_heading is always a structure boundary.
            _cid = chunk.get('clause_id') or ''
            _found_type = chunk.get('_found_type') or ''
            _is_cfr_start = _found_type in ('cfr_section_heading', 'cfr_part_heading')
            _is_sequence_start = _is_cfr_start or bool(
                re.match(r'^(?:article\s+1\b|1\.?$|1\.\d)', _cid, re.IGNORECASE)
            )
            if chunk.get('text_type') == 'heading' and _is_sequence_start:
                boundary_idx = idx
                break

        # Tag everything before the boundary (or within page cap if no boundary found)
        for idx, chunk in enumerate(raw_chunks):
            if boundary_idx is not None and idx >= boundary_idx:
                break
            if chunk.get('text_type') in SKIP_TYPES:
                continue
            if _first_page(chunk) > prelude_page_cap:
                break
            chunk['text_type'] = 'prelude'
            chunk['clause_id'] = None   # prelude chunks carry no clause identity
            chunk['level']     = None

        return raw_chunks

    # ─────────────────────────────────────────────────────────────────────────

    def _split_multi_heading_blocks(self, raw_chunks, regex_map):
        """
        STEP 1b: Split blocks that contain multiple heading patterns within their
        content. E.g., a PaddleOCR block with content:
            "143\nUniform provisions...\n144\nUniform provisions concerning:"
        is split into two separate chunks so each regulation number gets its own
        clause_id when _annotate_clause_ids runs in STEP 2.

        Only bare-integer lines (standalone regulation numbers) and roman-numeral
        lines that appear mid-block (after other content) are used as split points,
        to avoid false splits on numbers embedded in running text.
        """
        _bare_int   = re.compile(r'^\s*\d{1,3}\s*$')
        _ml_heading = regex_map.get('multi_level_heading')
        _ru_heading = regex_map.get('roman_upper_heading')
        _rua_heading = regex_map.get('roman_upper_alpha_heading')
        # CFR clause with optional year suffix, alone on its line: "86.135–90", "86.141"
        _cfr_year   = re.compile(r'^\s*\d{1,3}\.\d{1,3}(?:\.\d+)*(?:[–\-—]\d{2,4})?\s*$')

        result = []
        for chunk in raw_chunks:
            if chunk.get('text_type') in ('table', 'figure', 'toc', 'header', 'footer', 'footnote', 'vision_footnote'):
                result.append(chunk)
                continue

            content = chunk.get('content_verbatim', '')
            lines   = content.splitlines()
            if len(lines) <= 1:
                result.append(chunk)
                continue

            # Detect CFR-style section listing: block has ≥2 bare CFR-number lines.
            # In this format each entry is two lines: "86.135–90\nDynamometer procedure".
            # We split at every CFR-number line so each entry becomes its own chunk.
            cfr_number_indices = [
                i for i, ln in enumerate(lines)
                if _cfr_year.match(ln.strip()) and ln.strip()
            ]
            if len(cfr_number_indices) >= 2:
                split_at = cfr_number_indices
                split_at.append(len(lines))
                # Emit any lines before the first CFR number as a header chunk
                if split_at[0] > 0:
                    header_content = '\n'.join(lines[:split_at[0]]).strip()
                    if header_content:
                        hdr_chunk = dict(chunk)
                        hdr_chunk['content_verbatim'] = header_content
                        hdr_chunk['clause_id'] = None
                        hdr_chunk['title'] = "NIL"
                        hdr_chunk['level'] = None
                        result.append(hdr_chunk)
                for k in range(len(split_at) - 1):
                    sub_lines   = lines[split_at[k]:split_at[k + 1]]
                    sub_content = '\n'.join(sub_lines).strip()
                    if not sub_content:
                        continue
                    new_chunk = dict(chunk)
                    new_chunk['block_bboxes']     = {pg: list(bbs) for pg, bbs in chunk.get('block_bboxes', {}).items()}
                    new_chunk['content_words']    = list(chunk.get('content_words', []))
                    new_chunk['content_verbatim'] = sub_content
                    new_chunk['clause_id']        = None
                    new_chunk['title']            = "NIL"
                    new_chunk['level']            = None
                    result.append(new_chunk)
                continue

            # Find line indices where a new heading begins (not the very first line)
            split_at = [0]
            for i in range(1, len(lines)):
                ln = lines[i].strip()
                if not ln:
                    continue
                is_split = False
                if _bare_int.match(ln):
                    is_split = True
                elif _ml_heading and _ml_heading.match(ln):
                    m = _ml_heading.match(ln)
                    if m and m.group(1).strip().isdigit():
                        is_split = True
                elif _ru_heading and _ru_heading.match(ln):
                    is_split = True
                elif _rua_heading and _rua_heading.match(ln):
                    is_split = True
                elif re.match(r'^S\d+(?:\.\d+)*\.?\s', ln):
                    # FMVSS S-clause lines (e.g. "S2.  Application.", "S4.1  Frequency.")
                    # embedded inside a paragraph_title block must become their own chunks.
                    is_split = True
                if is_split:
                    split_at.append(i)

            if len(split_at) == 1:
                result.append(chunk)
                continue

            # Produce sub-chunks; each inherits everything from parent
            # but gets its own content slice (clause_id/level will be
            # re-derived by _annotate_clause_ids in STEP 2).
            split_at.append(len(lines))
            for k in range(len(split_at) - 1):
                sub_lines   = lines[split_at[k]:split_at[k + 1]]
                sub_content = '\n'.join(sub_lines).strip()
                if not sub_content:
                    continue
                new_chunk = dict(chunk)
                new_chunk['block_bboxes']     = {pg: list(bbs) for pg, bbs in chunk.get('block_bboxes', {}).items()}
                new_chunk['content_words']    = list(chunk.get('content_words', []))
                new_chunk['content_verbatim'] = sub_content
                new_chunk['clause_id']        = None
                new_chunk['title']            = "NIL"
                new_chunk['level']            = None
                result.append(new_chunk)

        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _merge_by_left_alignment(self, raw_chunks):
        """
        STEP 3: For each non-clause block, compare its left x-coordinate to the
        most recent clause chunk. If within ALIGNMENT_MERGE_TOLERANCE_PTS, merge
        the block's content into that clause chunk. Otherwise keep standalone.
        Tables, figures, toc blocks, and PaddleOCR-labelled headings are never
        merged regardless of position.
        """
        result            = []
        last_clause_chunk = None

        for chunk in raw_chunks:
            cid = chunk.get('clause_id')

            # ── Clause chunk: always keep, update pointer ──────────────────
            if cid is not None:
                result.append(chunk)
                last_clause_chunk = chunk
                continue

            # ── Never merge these types ────────────────────────────────────
            if chunk.get('text_type') in ('table', 'figure', 'figure_title', 'image_caption', 'toc', 'prelude', 'header', 'footer', 'footnote', 'vision_footnote'):
                result.append(chunk)
                continue

            # ── PaddleOCR structural heading without a regex match ─────────
            # Also update last_clause_chunk so body text after this heading
            # anchors to it rather than the previous numbered clause.
            if chunk.get('_block_label') in ('doc_title', 'paragraph_title', 'title'):
                result.append(chunk)
                last_clause_chunk = chunk
                continue

            # ── FMVSS-style clause block (e.g. "S4.1.1  Frequency.") ───────
            # Blocks whose first line starts with an S-numbered clause pattern
            # are always structural paragraphs and must never be absorbed into
            # a preceding clause even when they share the same left margin.
            # This prevents sub-clauses like S4.1.1 / S4.1.1.1 from being
            # merged into the parent S4 chunk, which would scramble reading order.
            _fmvss_first = next(
                (ln.strip() for ln in chunk.get('content_verbatim', '').splitlines()
                 if ln.strip()), ''
            )
            if re.match(r'^S\d+(?:\.\d+)*\.?\s', _fmvss_first):
                result.append(chunk)
                last_clause_chunk = chunk
                continue

            # ── No preceding clause chunk yet (preamble territory) ─────────
            if last_clause_chunk is None:
                result.append(chunk)
                continue

            chunk_x  = chunk.get('_block_left_x')
            clause_x = last_clause_chunk.get('_block_left_x')

            if chunk_x is None or clause_x is None:
                result.append(chunk)
                continue

            # ── Cross-page guard ───────────────────────────────────────────
            # Don't merge a chunk from a later page into a clause that is
            # entirely on an earlier page.  This prevents "Section" headers
            # (or other leading content from a rescued table-of-contents
            # block on the next page) from bleeding into the last clause
            # chunk of the previous page.
            # We allow the merge when the clause chunk already spans up to
            # the chunk's page (legitimate cross-page body text continuation).
            try:
                _chunk_min_pg = min(
                    int(p.strip()) for p in chunk.get('source_page', '').split(',')
                    if p.strip().isdigit()
                )
                _clause_max_pg = max(
                    int(p.strip()) for p in last_clause_chunk.get('source_page', '').split(',')
                    if p.strip().isdigit()
                )
                if _chunk_min_pg > _clause_max_pg:
                    result.append(chunk)
                    continue
            except ValueError:
                pass

            if abs(chunk_x - clause_x) <= ALIGNMENT_MERGE_TOLERANCE_PTS:
                # ── MERGE into last_clause_chunk ───────────────────────────
                sep = "\n" if last_clause_chunk['content_verbatim'].strip() else ""
                last_clause_chunk['content_verbatim'] += sep + chunk['content_verbatim']
                last_clause_chunk['content_words'].extend(
                    chunk.get('content_words', [])
                )
                for pg, bboxes in chunk.get('block_bboxes', {}).items():
                    pg_list = last_clause_chunk['block_bboxes'].setdefault(pg, [])
                    for bb in bboxes:
                        if bb not in pg_list:
                            pg_list.append(bb)
                existing = set(last_clause_chunk['source_page'].split(', '))
                new_pgs  = set(chunk['source_page'].split(', '))
                last_clause_chunk['source_page'] = ', '.join(
                    sorted(existing | new_pgs,
                           key=lambda x: int(x) if x.isdigit() else 0)
                )
            else:
                # ── Keep standalone ────────────────────────────────────────
                result.append(chunk)

        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _assign_parent_ids(self, chunks):
        """
        STEP 5: Stack-based parent ID assignment over the merged chunk list.
        Non-clause chunks (clause_id NIL/TOC/PRELUDE) inherit the current stack
        top as their parent without pushing to the stack.

        Each stack entry is (level, cid, x_start, is_centered, found_type).
        When a heading at the same nominal level has a left-margin (x_start)
        that is significantly greater than the previous same-level heading, it
        is treated as a child — BUT only when both blocks are left-aligned.
        Centered headings (e.g. "CHAPTER II") have a large x_start due to
        centering, not indentation, so the indentation check is skipped for them.
        """
        INDENT_CHILD_THRESHOLD = 10   # pts — minimum difference that signals a child indent
        COLUMN_JUMP_THRESHOLD  = 100  # pts — above this, x shift is a column break, not indentation
        # Heading types that represent a chapter/section grouping above articles.
        _CHAPTER_FOUND_TYPES = frozenset({
            'chapter_heading', 'roman_upper_heading',
            'roman_upper_alpha_heading', 'appendix_heading',
        })
        # List-item types whose pre-assigned level is only a rough guess.
        # The true level is resolved dynamically: one deeper than whatever the
        # pop loop found as the parent.  This prevents a hardcoded level=2 from
        # popping past a deeply-nested parent (e.g. 1.3.1.1 at level 4) and
        # also stops the pushed list item from incorrectly acting as an ancestor
        # of the next numbered heading (e.g. 1.3.2 at level 3).
        _LIST_FOUND_TYPES = frozenset({
            'numeric_paren_heading', 'alpha_paren_heading', 'alpha_bare_heading',
            'roman_lower_paren_heading', 'roman_lower_bare_heading', 'bullet_heading',
        })
        # Annotation labels that appear WITHIN clause body text (EXAMPLE, NOTE, etc.)
        # and must never act as section-title parents for the next numbered clause.
        _ANNOTATION_KEYWORDS = frozenset({
            'EXAMPLE', 'NOTE', 'WARNING', 'CAUTION', 'IMPORTANT', 'REMARK', 'INFORMATIVE',
        })
        _numeric_cid_re = re.compile(r'^\s*\d')

        # Stack entries: (level, cid, x_start, is_centered, found_type)
        parent_stack = [(0, "ROOT", 0.0, False, '')]
        # Tracks indentation of NIL (non-clause) text blocks so that a block
        # visually indented further right nests one level deeper.  Each entry
        # is (x_start, parent_id, level) of the most-recently-seen NIL block
        # at that indentation depth.  Resets whenever a real clause is pushed.
        _nil_indent_stack = []  # (x, parent_id, level)
        # Detects roman-numeral section titles used as top-level breaks
        # (e.g. "III DEVICES ON MOTOR VEHICLES...") that lack the trailing dot
        # required for roman_upper_heading and are promoted as doc_title_heading
        # in Pass 1b.  These must reset to ROOT, not nest under the prior clause.
        _roman_section_re = re.compile(
            r'^(I{1,3}|I?V|VI{0,3}|I?X|X(?:I{0,3}|I?V|VI{0,3}))\s+\S',
            re.IGNORECASE,
        )

        # Pages to trace parent-stack state for debugging parent_id issues.
        _STACK_DEBUG_PAGES = {3, 6, 7}

        for chunk in chunks:
            cid         = chunk.get('clause_id') or 'NIL'
            level       = chunk.get('level') or 0
            x_start     = float(chunk.get('_block_left_x') or 0)
            is_centered = bool(chunk.get('_block_centered', False))
            found_type  = chunk.get('_found_type') or ''

            # ── Parent-stack debug trace (pages 6–7) ─────────────────────────
            if self.session_logger:
                _chunk_pages = set(
                    str(p) for p in (chunk.get('block_bboxes') or {}).keys()
                )
                if _chunk_pages & {str(p) for p in _STACK_DEBUG_PAGES}:
                    _stack_summary = [(e[0], e[1], e[4]) for e in parent_stack]
                    _cv = (chunk.get('content_verbatim') or '').replace('\n', ' | ')[:80]
                    self.session_logger.info(
                        f"[STACK-DBG] cid={cid!r} found_type={found_type!r} "
                        f"text_type={chunk.get('text_type')!r} level={level} "
                        f"x={x_start} | stack={_stack_summary} | content={_cv!r}"
                    )
            # ─────────────────────────────────────────────────────────────────

            if chunk.get('text_type') in ('header', 'footer'):
                chunk['parent_id'] = 'ROOT'
                chunk['level'] = 0
                continue

            # Prelude chunks are outside the clause hierarchy — always ROOT, no stack push.
            # This also ensures the stack stays clean so the first post-prelude heading
            # correctly finds ROOT as its parent.
            if chunk.get('text_type') == 'prelude':
                chunk['parent_id'] = 'ROOT'
                chunk['level'] = 0
                continue

            if cid in ('NIL', 'TOC', 'PRELUDE'):
                # Figure/table label chunks (e.g. "Figure B94–1", "Table 3.2")
                # and figure_title/image_caption blocks always attach to the
                # current clause regardless of indentation position.
                _cv_strip = (chunk.get('content_verbatim') or '').strip()
                _is_fig_label = (
                    chunk.get('text_type') in ('figure_title', 'image_caption')
                    or bool(re.match(
                        r'(?i)^(figure|table|fig\.?)\s+[A-Z0-9][A-Z0-9\-–—\.]*\s*$',
                        _cv_strip
                    ))
                )
                if _is_fig_label:
                    chunk['parent_id'] = parent_stack[-1][1]
                    chunk['level']     = parent_stack[-1][0] + 1
                    continue  # do not push to nil_indent_stack
                # Pop NIL entries that are at the same or greater indentation
                # depth — we only keep entries that are strictly shallower.
                while _nil_indent_stack and x_start - _nil_indent_stack[-1][0] <= INDENT_CHILD_THRESHOLD:
                    _nil_indent_stack.pop()
                if _nil_indent_stack:
                    # More indented than the last NIL block — nest one deeper.
                    nil_parent_id = _nil_indent_stack[-1][1]
                    nil_level     = _nil_indent_stack[-1][2] + 1
                else:
                    # At or shallower than the clause stack — inherit normally.
                    nil_parent_id = parent_stack[-1][1]
                    nil_level     = parent_stack[-1][0] + 1
                chunk['parent_id'] = nil_parent_id
                chunk['level']     = nil_level
                _nil_indent_stack.append((x_start, nil_parent_id, nil_level))
                continue

            # Reset stack at annex/appendix/addendum boundary — these are new
            # top-level section breaks that start fresh regardless of prior depth.
            if isinstance(cid, str) and re.match(r'^(ANNEX|APPENDIX|ANNEXES|ANNEXURE|ADDENDUM)\b', cid.upper()):
                parent_stack = [(0, "ROOT", 0.0, False, '')]

            # ── Special case: article_heading ────────────────────────────────
            # In hierarchical regulation documents (e.g. EU regulations), articles
            # sit BELOW chapter headings.  Because article_heading is assigned
            # level=0 (so articles appear at the top of the clause tree in documents
            # that have no chapters), the generic level-pop loop would pop chapter
            # entries off the stack and give the article parent_id=ROOT.
            #
            # Fix: walk the stack from the top looking for the nearest
            # chapter-type heading.  If found, trim the stack to that point and
            # assign the article as its child at chapter_level+1.  If no chapter
            # exists, fall through to the standard pop loop which will land at ROOT.
            if found_type == 'article_heading':
                # If a previous article_heading is already on the stack, this new
                # article is a sibling — trim the stack back to just before that
                # article so any numbered list items nested inside it (e.g. "1.", "2.")
                # are not mistaken for containing sections.
                for i in range(len(parent_stack) - 1, -1, -1):
                    if parent_stack[i][4] == 'article_heading':
                        parent_stack = parent_stack[:i]
                        break

                # When the stack top is a numbered section heading (e.g. "1. VEHICLE
                # STEERING SYSTEMS"), articles nest directly under that section rather
                # than searching upward for a chapter — the section is their container.
                if parent_stack[-1][4] == 'multi_level_heading':
                    section_level = parent_stack[-1][0]
                    article_level = section_level + 1
                    chunk['parent_id'] = parent_stack[-1][1]
                    chunk['level'] = article_level
                    parent_stack.append((article_level, cid, x_start, is_centered, found_type))
                    _nil_indent_stack.clear()
                    continue
                chapter_idx = None
                for i in range(len(parent_stack) - 1, -1, -1):
                    if parent_stack[i][4] in _CHAPTER_FOUND_TYPES or parent_stack[i][4] == 'multi_level_heading':
                        chapter_idx = i
                        break
                if chapter_idx is not None:
                    # Trim stack to immediately after the chapter/section entry
                    parent_stack = parent_stack[:chapter_idx + 1]
                    chapter_level = parent_stack[-1][0]
                    article_level = chapter_level + 1
                    chunk['parent_id'] = parent_stack[-1][1]
                    chunk['level'] = article_level
                    parent_stack.append((article_level, cid, x_start, is_centered, found_type))
                    _nil_indent_stack.clear()
                    continue
                # No chapter/section in stack — fall through to standard pop logic below

            # ── Special case: cfr_section_heading ────────────────────────────
            # CFR sections (§86.101, §86.102 …) must nest under the nearest
            # SUBPART (a doc_title_heading whose cid starts with "SUBPART") or,
            # if no SUBPART is present, under the nearest cfr_part_heading.
            # Without this, §86.101 arrives at level=1, pops SUBPART B
            # (level=2) off the stack, then pops PART 86 (level=1) as well,
            # landing at ROOT instead of under the subpart.
            # Sibling cfr_section_headings are also popped so they don't act
            # as ancestors of one another.
            if found_type == 'cfr_section_heading':
                container_idx = None
                for i in range(len(parent_stack) - 1, -1, -1):
                    stk_ft  = parent_stack[i][4]
                    stk_cid = str(parent_stack[i][1])
                    if stk_ft == 'doc_title_heading' and re.match(r'(?i)^SUBPART\b', stk_cid):
                        container_idx = i
                        break
                    if stk_ft == 'cfr_part_heading':
                        container_idx = i
                        break
                if container_idx is not None:
                    parent_stack = parent_stack[:container_idx + 1]
                    container_level = parent_stack[-1][0]
                    section_level   = container_level + 1
                    chunk['parent_id'] = parent_stack[-1][1]
                    chunk['level']     = section_level
                    parent_stack.append((section_level, cid, x_start, is_centered, found_type))
                    _nil_indent_stack.clear()
                    continue
                # No SUBPART/PART on stack — fall through to standard pop logic

            # ── Special case: section_general_heading ────────────────────────
            # Sections (e.g. "SECTION A") sit BELOW part/chapter headings.
            # Without this, both cfr_part_heading and section_general_heading
            # default to level=1, causing SECTION A to pop PART 2 off the stack
            # and incorrectly receive parent_id=ROOT.
            if found_type == 'section_general_heading':
                part_idx = None
                for i in range(len(parent_stack) - 1, -1, -1):
                    if parent_stack[i][4] in ('cfr_part_heading', 'chapter_heading'):
                        part_idx = i
                        break
                if part_idx is not None:
                    parent_stack = parent_stack[:part_idx + 1]
                    part_level = parent_stack[-1][0]
                    section_level = part_level + 1
                    chunk['parent_id'] = parent_stack[-1][1]
                    chunk['level'] = section_level
                    parent_stack.append((section_level, cid, x_start, is_centered, found_type))
                    _nil_indent_stack.clear()
                    continue
                # No part/chapter in stack — fall through to standard pop logic

            # ── doc_title fast path ──────────────────────────────────────────
            # Visual title blocks (Annex header sequences like PART 1 →
            # INFORMATION DOCUMENT → MODEL) are labels for the section they
            # appear in.  They must NEVER pop the stack — just nest one level
            # deeper than whatever heading is currently on top.
            #
            # Exception: paddle-labeled headings whose clause_id begins with a
            # bare roman numeral (e.g. "III DEVICES ON MOTOR VEHICLES...") are
            # top-level section breaks, not annex title sequences.  They lack
            # the trailing dot required for roman_upper_heading detection in
            # Pass 1 and therefore end up as doc_title_heading in Pass 1b.
            # Reset the stack to ROOT so they always parent to ROOT.
            if found_type == 'doc_title_heading':
                if _roman_section_re.match(str(cid)):
                    parent_stack = [(0, "ROOT", 0.0, False, '')]
                    chunk['level'] = 1
                    chunk['parent_id'] = 'ROOT'
                    parent_stack.append((1, cid, x_start, is_centered, found_type))
                    _nil_indent_stack.clear()
                    continue
                # CFR Subpart headings (e.g. "SUBPART B", "SUBPART A—GENERAL")
                # sit directly under cfr_part_heading (e.g. "PART 86").
                # Search the stack for the nearest cfr_part_heading and nest under
                # it; if none found, fall back to ROOT so _chain_prelude_parents
                # won't wrongly assign a prelude parent.
                if re.match(r'(?i)^SUBPART\b', str(cid)):
                    part_idx = None
                    for i in range(len(parent_stack) - 1, -1, -1):
                        if parent_stack[i][4] == 'cfr_part_heading':
                            part_idx = i
                            break
                    if part_idx is not None:
                        parent_stack = parent_stack[:part_idx + 1]
                        part_level = parent_stack[-1][0]
                        subpart_level = part_level + 1
                        chunk['parent_id'] = parent_stack[-1][1]
                        chunk['level'] = subpart_level
                        parent_stack.append((subpart_level, cid, x_start, is_centered, found_type))
                    else:
                        parent_stack = [(0, "ROOT", 0.0, False, '')]
                        chunk['level'] = 1
                        chunk['parent_id'] = 'ROOT'
                        parent_stack.append((1, cid, x_start, is_centered, found_type))
                    _nil_indent_stack.clear()
                    continue
                # Pop annotation labels (EXAMPLE, NOTE, etc.) — they are subordinate
                # labels within a clause body, not structural section anchors.
                while (len(parent_stack) > 1 and
                       str(parent_stack[-1][1]).strip().split()[0].upper()
                       in _ANNOTATION_KEYWORDS):
                    parent_stack.pop()
                # If this block looks like a numbered section (e.g. "2 Normative
                # references"), pop numbered-section doc_title siblings so it lands
                # as a sibling rather than a grandchild of the previous section.
                if _numeric_cid_re.match(str(cid)):
                    while (len(parent_stack) > 1 and
                           parent_stack[-1][4] == 'doc_title_heading' and
                           _numeric_cid_re.match(str(parent_stack[-1][1]))):
                        parent_stack.pop()
                level = parent_stack[-1][0] + 1
                chunk['level'] = level
                chunk['parent_id'] = parent_stack[-1][1]
                parent_stack.append((level, cid, x_start, is_centered, found_type))
                continue

            # ── List-item fast path ───────────────────────────────────────────
            # List items (alpha/numeric/bullet) must NEVER pop numbered headings
            # off the stack.  Their pre-assigned level (hardcoded 2 or 3) is
            # meaningless when the parent heading is deep (e.g. 1.3.1.1 at
            # level 4).  Instead: pop only sibling list items, stop the moment
            # we hit any non-list entry (that entry is the true parent), then
            # compute level dynamically as parent_level + 1.
            if found_type in _LIST_FOUND_TYPES:
                while len(parent_stack) > 1 and parent_stack[-1][1] != "ROOT":
                    stk_level, stk_cid, stk_x, stk_cen, stk_ft = parent_stack[-1]
                    if stk_ft not in _LIST_FOUND_TYPES:
                        # doc_title_heading entries (e.g. repeated page watermarks like
                        # "TITLE: …") are metadata labels, not true structural parents
                        # for list items.  Pop them so the list item finds the real
                        # section heading (cfr_section_heading, multi_level_heading, …)
                        # further up the stack.
                        if stk_ft == 'doc_title_heading':
                            parent_stack.pop()
                            continue
                        break  # reached a numbered heading — this is the parent
                    # Indentation check: only meaningful when both blocks are
                    # left-aligned (not centered on the page).
                    if (not is_centered and not stk_cen
                            and x_start - stk_x > INDENT_CHILD_THRESHOLD):
                        break  # more indented → nest under the list item above
                    parent_stack.pop()
                level = parent_stack[-1][0] + 1
                chunk['level'] = level
                chunk['parent_id'] = parent_stack[-1][1]
                parent_stack.append((level, cid, x_start, is_centered, found_type))
                continue

            # Label headings (clause_id ends with ":", e.g. "TITLE:", "DATE:")
            # are metadata stamps or running-header labels that should be
            # assigned a structural parent for output but must not evict real
            # CFR/numbered entries from the parent stack.  Save the stack
            # before the pop loop and restore it afterward so sibling clauses
            # on the same page continue to find the correct parent.
            _is_label_heading = str(cid).strip().endswith(':')
            _saved_stack = list(parent_stack) if _is_label_heading else None

            # Pop entries at same or deeper level, but keep a same-level entry
            # when the new heading is truly indented relative to it — i.e. both
            # blocks are left-aligned, not centered (centered headings like
            # "CHAPTER II" have a large x_start from centering, not indentation).
            while len(parent_stack) > 1 and parent_stack[-1][1] != "ROOT":
                stk_level, stk_cid, stk_x, stk_cen, stk_ft = parent_stack[-1]
                # Metadata label headings whose clause_id ends with ":"
                # (e.g. "TITLE:", "DATE:") are section labels or running-header
                # stamps that may parent indented body text but must never parent
                # numbered CFR/structured clauses.  Pop them before the level
                # check so the real structural ancestor is found instead.
                if (str(stk_cid).strip().endswith(':')
                        and _numeric_cid_re.match(str(cid))):
                    parent_stack.pop()
                    continue
                if stk_level < level:
                    break  # found a genuine parent
                if (stk_level == level
                        and not is_centered and not stk_cen
                        and INDENT_CHILD_THRESHOLD < x_start - stk_x <= COLUMN_JUMP_THRESHOLD):
                    break  # same level, both left-aligned, and indented → treat as parent
                    # (gap > COLUMN_JUMP_THRESHOLD means a multi-column layout shift, not nesting)
                # A Pass-1b all-caps/bold title (no _found_type) acts as a title
                # parent for subsequent pattern-matched numbered headings at the
                # same nominal level (e.g. "1." under "SUBJECT MATTER").
                # Exception: known annotation labels (EXAMPLE, NOTE, etc.) are
                # subordinate body labels, not section-title parents, so they
                # must be popped rather than kept as parents.
                if stk_level == level and stk_ft == '' and found_type != '':
                    _stk_first_word = str(stk_cid).strip().split()[0].upper() if stk_cid else ''
                    if _stk_first_word not in _ANNOTATION_KEYWORDS:
                        # Cross-reference IDs like "2553 (Part 1) :" start with a
                        # digit but are NOT clean section numbers (digits+dots only).
                        # They must not act as title-parents for the next numbered
                        # section — only clean section numbers (e.g. "2") or
                        # non-numeric titles (e.g. "SCOPE") qualify.
                        stk_cid_str = str(stk_cid).strip() if stk_cid else ''
                        if not _numeric_cid_re.match(stk_cid_str) or re.match(r'^\d[\d.]*$', stk_cid_str):
                            break
                # A Pass-1 structured heading (e.g. "PART 2") stays as parent
                # for a Pass-1b all-caps block (e.g. "MODEL") at the same level —
                # the unstructured block should nest under the structured one.
                if stk_level == level and stk_ft != '' and found_type == '':
                    break
                # doc_title blocks are sticky anchors for headings that are truly
                # nested under them (incoming level > doc_title level).  Annex
                # title sequences like TYPE-APPROVAL → PART 1 → INFORMATION
                # DOCUMENT must nest under their preceding title label.
                # Exception: if the doc_title was pushed deeper than the incoming
                # heading (e.g. a page watermark "TITLE:" pushed at level 3 while
                # the next CFR section is at level 2), pop it so the section finds
                # its real structural parent (e.g. the PART heading at level 1).
                if stk_ft == 'doc_title_heading' and stk_level <= level:
                    break
                # An article_heading is a parent for numbered clause headings
                # (e.g. "1.", "2." inside ARTICLE 6) even when the numbered
                # heading's raw level is less than the article's assigned level.
                if stk_ft == 'article_heading' and _numeric_cid_re.match(str(cid)):
                    break
                parent_stack.pop()

            chunk['parent_id'] = parent_stack[-1][1]
            # ── Parent-stack debug: log final assignment for pages 6–7 ───────
            if self.session_logger:
                _chunk_pages = set(
                    str(p) for p in (chunk.get('block_bboxes') or {}).keys()
                )
                if _chunk_pages & {str(p) for p in _STACK_DEBUG_PAGES}:
                    self.session_logger.info(
                        f"[STACK-DBG] → assigned parent_id={chunk['parent_id']!r} "
                        f"to cid={cid!r}"
                    )
            # ─────────────────────────────────────────────────────────────────
            if _saved_stack is not None:
                # Label heading: restore the stack to its pre-pop state so
                # the label doesn't evict structural CFR entries that sibling
                # clauses on the same page still need to find.
                parent_stack[:] = _saved_stack
                continue
            parent_stack.append((level, cid, x_start, is_centered, found_type))
            # A real clause heading resets the NIL indent stack so subsequent
            # text blocks start fresh relative to this heading's position.
            _nil_indent_stack.clear()

        return chunks

    # ─────────────────────────────────────────────────────────────────────────

    def _assign_vision_footnote_parents(self, chunks):
        """
        STEP 5b: For every vision_footnote chunk, set parent_id to the same
        parent_id carried by the nearest preceding table chunk on the same page
        (i.e. the clause that owns the table, e.g. "3.3").  This makes the
        vision_footnote a sibling of the table under the same clause rather than
        a child of the raw table block.

        If no table chunk is found on that page, fall back to the parent_id of
        the nearest preceding non-edge chunk (the chunk that will receive any
        adjacent base64 image attachment).

        This must run AFTER _assign_parent_ids (Step 5) so that the table/figure
        chunk's parent_id is already resolved before we inherit it here.
        """
        EDGE_TYPES = {'header', 'footer', 'footnote', 'vision_footnote'}

        for i, chunk in enumerate(chunks):
            if chunk.get('text_type') != 'vision_footnote':
                continue

            chunk_page = chunk.get('source_page', '')
            inherited_parent = None

            # First pass: nearest preceding table chunk on same page — inherit
            # its parent_id (the clause that owns the table, e.g. "3.3").
            for j in range(i - 1, -1, -1):
                candidate = chunks[j]
                if candidate.get('source_page', '') != chunk_page:
                    continue
                if candidate.get('text_type') == 'table':
                    inherited_parent = candidate.get('parent_id')
                    break

            # Second pass: if no table found, inherit parent_id from the nearest
            # preceding non-edge chunk (the image attachment target).
            if inherited_parent is None:
                for j in range(i - 1, -1, -1):
                    candidate = chunks[j]
                    if candidate.get('source_page', '') != chunk_page:
                        continue
                    if candidate.get('text_type') not in EDGE_TYPES:
                        inherited_parent = candidate.get('parent_id')
                        break

            if inherited_parent is not None:
                chunk['parent_id'] = inherited_parent

        return chunks

    # ─────────────────────────────────────────────────────────────────────────

    def _detect_annex_appendix(self, chunks):
        """
        STEP 6: Walk all chunks in order and propagate the running annex/appendix
        context to every chunk's 'annex_appendix' field. The context is updated
        whenever a chunk's clause_id starts with ANNEX or APPENDIX.
        """
        current_annex_context = "NIL"

        for chunk in chunks:
            # Prelude chunks are before any annex context exists — keep them NIL.
            if chunk.get('text_type') == 'prelude':
                chunk['annex_appendix'] = 'NIL'
                continue
            cid = chunk.get('clause_id') or ''
            if (isinstance(cid, str)
                    and len(cid) > 5
                    and cid.upper().startswith(("ANNEX ", "APPENDIX "))):
                current_annex_context = cid
            chunk['annex_appendix'] = current_annex_context

        return chunks

    # ─────────────────────────────────────────────────────────────────────────

    def hybrid_extract_and_structure(self, doc, num_pages, paddle_model, model_type="vl",
                                      progress_callback=None, status_callback=None,
                                      checkpoint_path=None, resume_from_page=0, page_start=0):
        # ── Initialize accumulators ───────────────────────────────────────────
        all_chunks      = []    # finalized chunks (populated after post-processing)
        raw_chunks      = []    # block-level initial chunks built page-by-page
        all_page_images = []
        chunk_counter   = 1
        image_counter   = 1

        metrics = {
            "total_words": 0, "kept_words": 0, "dropped_words": 0,
            "debug_images": {}, "detected_tables": defaultdict(list),
            "removed_header_lines": 0, "removed_footer_lines": 0,
            "removed_edge_line_samples": []
        }

        # ── Checkpoint restore ────────────────────────────────────────────────
        if checkpoint_path and resume_from_page > 0:
            ckpt = self._load_checkpoint(checkpoint_path)
            if ckpt:
                all_chunks      = ckpt.get("all_chunks", [])
                raw_chunks      = ckpt.get("raw_chunks", [])
                all_page_images = ckpt.get("all_page_images", [])
                s               = ckpt.get("state", {})
                chunk_counter   = s.get("chunk_counter", 1)
                image_counter   = s.get("image_counter", 1)
                sm = ckpt.get("metrics", {})
                metrics["total_words"]               = sm.get("total_words", 0)
                metrics["kept_words"]                = sm.get("kept_words", 0)
                metrics["dropped_words"]             = sm.get("dropped_words", 0)
                metrics["removed_header_lines"]      = sm.get("removed_header_lines", 0)
                metrics["removed_footer_lines"]      = sm.get("removed_footer_lines", 0)
                metrics["removed_edge_line_samples"] = sm.get("removed_edge_line_samples", [])
                for pg, rects in sm.get("detected_tables", {}).items():
                    metrics["detected_tables"][pg] = rects
                for pg, rects in sm.get("detected_columnar_content", {}).items():
                    metrics.setdefault("detected_columnar_content",
                                       defaultdict(list))[pg] = rects
                limit_display = min(len(doc), num_pages)
                st.info(
                    f"\u2705 Checkpoint restored \u2014 resuming from page "
                    f"**{resume_from_page + 1}** of {limit_display}."
                )
                if self.session_logger:
                    self.session_logger.info(
                        f"[CHECKPOINT RESTORED] resuming from page={resume_from_page + 1} "
                        f"all_chunks={len(all_chunks)} raw_chunks={len(raw_chunks)} "
                        f"chunk_counter={chunk_counter}"
                    )
        # ─────────────────────────────────────────────────────────────────────

        # ── Pre-compile regex patterns (used by _annotate_clause_ids) ─────────
        patterns = self.get_compliance_patterns()
        patterns['content_start_heading'] = (
            r'^\s*(INTRODUCTION|FOREWORD|PREAMBLE|PURPOSE|SCOPE)(?![a-z])'
        )
        regex_map = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}
        heading_priority = [
            'toc_start_heading', 'preamble_heading', 'content_start_heading',
            'appendix_heading', 'chapter_heading', 'section_general_heading',
            'cfr_part_heading', 'cfr_section_heading', 'fmvss_paragraph',
            'article_heading', 'roman_upper_heading', 'roman_upper_alpha_heading',
            'cfr_clause_year_heading', 'multi_level_heading',
            'numeric_paren_heading', 'alpha_paren_heading',
            'roman_lower_paren_heading', 'roman_lower_bare_heading',
            'alpha_bare_heading', 'bullet_heading',
            'ece_regulation_heading',
        ]
        month_pattern = re.compile(
            r'(?i)\b(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\b'
        )
        # ─────────────────────────────────────────────────────────────────────

        limit = min(len(doc), num_pages)
        header_cutoff, footer_cutoff, repeated_headers, repeated_footers = \
            self.detect_header_footer_zones(doc, sample_pages=limit)
        if self.session_logger:
            self.session_logger.info(
                f"[LAYOUT] header_cutoff={header_cutoff:.1f} "
                f"footer_cutoff={footer_cutoff:.1f} "
                f"repeated_headers={len(repeated_headers)} "
                f"repeated_footers={len(repeated_footers)}"
            )

        # ── Start isolated GPU worker subprocess (reuse across PDFs if alive) ──
        _cached = st.session_state.get("_paddle_worker")
        _reusing = (
            _cached is not None
            and _cached.get("model_type") == model_type
            and _cached["proc"].is_alive()
        )
        if _reusing:
            _worker_proc = _cached["proc"]
            _task_q      = _cached["task_q"]
            _result_q    = _cached["result_q"]
            if self.session_logger:
                self.session_logger.info(f"[GPU WORKER] Reusing existing worker pid={_worker_proc.pid}")
        else:
            if status_callback:
                status_callback("Loading Vision Model into GPU (may take 1–2 min)...")
            if _cached is not None:
                _stop_paddle_worker(_cached["proc"], _cached["task_q"], self.session_logger)
            try:
                _worker_proc, _task_q, _result_q = _start_paddle_worker(
                    model_type, logger=self.session_logger
                )
                st.session_state["_paddle_worker"] = {
                    "proc": _worker_proc, "task_q": _task_q,
                    "result_q": _result_q, "model_type": model_type,
                }
            except Exception as _we:
                if self.session_logger:
                    self.session_logger.error(f"[GPU WORKER START FAILED] {_we}")
                st.error(f"\u26d4 Failed to start GPU worker subprocess: {_we}")
                return [], metrics
        # ─────────────────────────────────────────────────────────────────────

        # When a checkpoint resume is active it takes precedence; otherwise use page_start.
        loop_start = resume_from_page if resume_from_page > 0 else page_start

        try:
          for p_idx in range(loop_start, limit):
            if progress_callback:
                progress_callback((p_idx - loop_start + 1) / (limit - loop_start) if limit > loop_start else 1.0)

            p_num       = p_idx + 1
            page        = doc[p_idx]
            _page_t0    = time.time()
            _raw_before = len(raw_chunks)

            # Count raw words (true baseline) and warn once if non-Latin script
            # is detected while the OCR model is English-only.
            try:
                raw_words = page.get_text("words")
                metrics['total_words'] += len(raw_words)
                if (model_type != "vl"
                        and OCR_LANGUAGE == "en"
                        and not getattr(self, '_non_latin_warned', False)):
                    _page_text_sample = page.get_text("text")[:500]
                    if self._detect_non_latin_script(_page_text_sample):
                        self._non_latin_warned = True
                        st.warning(
                            "Non-Latin script detected (Hindi, Arabic, CJK, etc.). "
                            "The current OCR model is English-only — non-Latin text "
                            "may be garbled. To fix, set `OCR_LANGUAGE` in the "
                            "CONFIGURATION section at the top of extracter.py "
                            "(e.g. `\"hi\"` for Hindi, `\"ml\"` for multilingual)."
                        )
                        if self.session_logger:
                            self.session_logger.warning(
                                f"[PAGE {p_num}] Non-Latin script detected; OCR_LANGUAGE={OCR_LANGUAGE!r}"
                            )
            except Exception:
                pass

            # Render page to image
            zoom = 2.0
            pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            if pix.n < 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            if pix.n == 4:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            img_bgr = np.ascontiguousarray(img_bgr)
            del pix, img_data  # free raw pixel buffer; img_bgr is an independent copy

            # PaddleOCR predict via isolated subprocess
            try:
                layout_results, _worker_proc, _task_q, _result_q = _predict_via_worker(
                    _worker_proc, _task_q, _result_q,
                    img_bgr, model_type,
                    logger=self.session_logger, page_num=p_num,
                )
            except Exception as _predict_err:
                if self.session_logger:
                    self.session_logger.error(
                        f"[PAGE {p_num} SKIP] Subprocess predict failed: {_predict_err}"
                    )
                st.warning(
                    f"\u26a0\ufe0f **Page {p_num} skipped** \u2014 GPU worker failed "
                    f"after all retries.  \n`{_predict_err}`  \n"
                    f"Processing continues on remaining pages."
                )
                gc.collect()
                continue

            all_blocks   = self._parse_paddle_output(layout_results, model_type)
            # Keep all blocks — header/footer/footnote are now included and tagged.
            # Exclude bare page-number noise (label 'number' / 'page_number').
            _excluded = {'number', 'page_number'}
            valid_blocks = [b for b in all_blocks if b['label'].lower() not in _excluded]
            self._assign_synthetic_order(valid_blocks)
            valid_blocks.sort(key=self._get_sorting_key)

            if self.session_logger and p_num in TOC_DEBUG_PAGES:
                self.session_logger.info(
                    f"[TOCDBG][raw-blocks][page {p_num}] count={len(valid_blocks)}"
                )
                for _bi, _b in enumerate(valid_blocks):
                    _ocr = re.sub(r'\s+', ' ', str(_b.get('ocr_text', '') or '').strip())
                    if len(_ocr) > 180:
                        _ocr = _ocr[:180] + "..."
                    self.session_logger.info(
                        f"[TOCDBG][raw-blocks][page {p_num}] idx={_bi} "
                        f"label={_b.get('label', '')!r} forced_zone={_b.get('_forced_zone')} "
                        f"bbox={_b.get('bbox')} ocr={_ocr!r}"
                    )

            viz_image = img_bgr.copy()
            scale_x   = page.rect.width  / img_bgr.shape[1]
            scale_y   = page.rect.height / img_bgr.shape[0]

            # ── Block-level header/footer zone tagger ─────────────────────
            # Blocks labeled 'text' that sit in the header/footer margin are
            # stamped with _forced_zone so _ocr_and_create_block_chunks can
            # tag them correctly instead of discarding them.
            for _b in valid_blocks:
                _zone = self._block_is_in_header_footer_zone(
                    _b, scale_x, scale_y,
                    header_cutoff, footer_cutoff,
                    repeated_headers, repeated_footers, page,
                )
                if _zone:
                    _b['_forced_zone'] = _zone

            table_counter     = 1
            page_image_bboxes = []

            # ── STEP 1: One initial chunk per PaddleOCR block ─────────────
            page_block_chunks, table_counter, image_counter, page_image_bboxes = \
                self._ocr_and_create_block_chunks(
                    valid_blocks, page, p_num,
                    viz_image, img_bgr, scale_x, scale_y,
                    header_cutoff, footer_cutoff,
                    repeated_headers, repeated_footers,
                    table_counter, image_counter, page_image_bboxes, metrics,
                )

            if self.session_logger and p_num in TOC_DEBUG_PAGES:
                self.session_logger.info(
                    f"[TOCDBG][page-block-chunks][page {p_num}] count={len(page_block_chunks)}"
                )
                for _ci, _c in enumerate(page_block_chunks):
                    _cv = re.sub(r'\s+', ' ', str(_c.get('content_verbatim', '') or '').replace('\n', ' | ').strip())
                    if len(_cv) > 220:
                        _cv = _cv[:220] + "..."
                    self.session_logger.info(
                        f"[TOCDBG][page-block-chunks][page {p_num}] idx={_ci} "
                        f"label={_c.get('_block_label', '')!r} text_type={_c.get('text_type', '')!r} "
                        f"bbox={(_c.get('block_bboxes', {}) or {}).get(str(p_num), [])} "
                        f"content={_cv!r}"
                    )
            raw_chunks.extend(page_block_chunks)
            # ─────────────────────────────────────────────────────────────

            # Image bbox merging + base64 crop
            merged_img_bboxes = self._merge_image_bboxes(page_image_bboxes)
            for mbox in merged_img_bboxes:
                mx1, my1, mx2, my2 = map(int, mbox)
                mx1, my1 = max(0, mx1), max(0, my1)
                mx2, my2 = (
                    min(img_bgr.shape[1], mx2),
                    min(img_bgr.shape[0], my2),
                )
                crop = img_bgr[my1:my2, mx1:mx2]
                if crop.size > 0:
                    try:
                        _, buf = cv2.imencode(".png", crop)
                        b64    = base64.b64encode(buf.tobytes()).decode("utf-8")
                        y_center = (my1 + my2) / 2.0 * scale_y
                        all_page_images.append({
                            "page":     p_num,
                            "y_center": y_center,
                            "y0":       my1 * scale_y,
                            "y1":       my2 * scale_y,
                            "image": {
                                "page":         p_num,
                                "image_number": image_counter,
                                "type":         "figure",
                                "mime_type":    "image/png",
                                "data":         b64,
                                "bbox":         {str(p_num): [[
                                    round(mx1 * scale_x, 1),
                                    round(my1 * scale_y, 1),
                                    round(mx2 * scale_x, 1),
                                    round(my2 * scale_y, 1),
                                ]]},
                            },
                        })
                        image_counter += 1
                    except Exception:
                        pass

            # Debug image encode + free large arrays (cap at 50 pages to limit RAM)
            success, buffer = cv2.imencode(".jpg", viz_image)
            if success and len(metrics['debug_images']) < 50:
                metrics['debug_images'][p_num] = buffer.tobytes()
            del viz_image, img_bgr

            # Per-page session log
            if self.session_logger:
                _label_counts = Counter(b['label'].lower() for b in all_blocks)
                _new_raw      = len(raw_chunks) - _raw_before
                self.session_logger.info(
                    f"[PAGE {p_num}/{limit}] t={time.time() - _page_t0:.1f}s "
                    f"blocks={len(all_blocks)} valid={len(valid_blocks)} "
                    f"labels={dict(_label_counts)} "
                    f"new_raw_chunks={_new_raw} total_raw={len(raw_chunks)}"
                )

            # ── Checkpoint every 10 pages ─────────────────────────────────
            if (p_idx + 1) % 10 == 0:
                gc.collect()
                if checkpoint_path:
                    self._save_checkpoint(
                        checkpoint_path,
                        p_idx + 1,
                        all_chunks, raw_chunks, all_page_images,
                        {
                            "chunk_counter": chunk_counter,
                            "image_counter": image_counter,
                        },
                        metrics,
                    )
                if self.session_logger:
                    self.session_logger.info(
                        f"[CHECKPOINT] page={p_num} raw_chunks={len(raw_chunks)} "
                        f"checkpoint={'saved' if checkpoint_path else 'skipped'}"
                    )
            # ─────────────────────────────────────────────────────────────

        except Exception as _loop_err:
            if self.session_logger:
                self.session_logger.error(
                    f"[SESSION INTERRUPTED] page={locals().get('p_num', '?')} "
                    f"reason={type(_loop_err).__name__}: {_loop_err}"
                )
            raise
        finally:
            # Keep the worker alive for reuse; update session_state in case it restarted
            if _worker_proc.is_alive():
                st.session_state["_paddle_worker"] = {
                    "proc": _worker_proc, "task_q": _task_q,
                    "result_q": _result_q, "model_type": model_type,
                }
            else:
                st.session_state.pop("_paddle_worker", None)

        # ── STEP 1b: Split blocks containing multiple embedded headings ─────────
        raw_chunks = self._split_multi_heading_blocks(raw_chunks, regex_map)
        self._log_toc_debug_snapshot("post-split", raw_chunks)

        # ── STEP 2: Clause ID annotation pass ────────────────────────────────
        raw_chunks = self._annotate_clause_ids(
            raw_chunks, regex_map, heading_priority, month_pattern
        )
        self._log_toc_debug_snapshot("post-annotate", raw_chunks)

        # ── STEP 2.5: Prelude annotation pass ────────────────────────────────
        raw_chunks = self._annotate_prelude(raw_chunks)
        self._log_toc_debug_snapshot("post-prelude", raw_chunks)

        # ── STEP 3: Merge non-clause blocks by left-x alignment ───────────────
        raw_chunks = self._merge_by_left_alignment(raw_chunks)
        self._log_toc_debug_snapshot("post-left-merge", raw_chunks)

        # ── DEBUG: log raw_chunks order for page 4 before ID assignment ─────────
        if self.session_logger:
            _pg4 = [c for c in raw_chunks if '4' in (c.get('block_bboxes') or {}).keys()]
            for _di, _dc in enumerate(_pg4):
                _cv = str(_dc.get('content_verbatim') or '').replace('\n', ' | ')[:60]
                _bb = list(((_dc.get('block_bboxes') or {}).get('4') or [[None]])[0])
                self.session_logger.info(
                    f"[CHUNK_ORDER_DEBUG][page4] pos={_di} "
                    f"text_type={_dc.get('text_type')!r} "
                    f"clause_id={_dc.get('clause_id')!r} "
                    f"bbox_y={_bb[1] if len(_bb)>1 else '?'} "
                    f"content={_cv!r}"
                )

        # ── Re-sort by (first_page, min_y) to fix any pipeline reordering ───────
        # Block-level sort runs early, but intermediate passes (e.g. clause
        # annotation, merging) can occasionally shift chunks out of page-reading
        # order.  Sorting here — right before IDs are locked in — guarantees
        # chunk_ids reflect correct top-to-bottom, page-by-page reading order.
        def _chunk_reading_order(chunk):
            bbs = chunk.get('block_bboxes') or {}
            pages = [int(p) for p in bbs.keys() if str(p).isdigit()]
            if not pages:
                return (9999, 9999.0)
            min_pg = min(pages)
            pg_boxes = bbs.get(str(min_pg), [])
            min_y = min((bb[1] for bb in pg_boxes if len(bb) > 1), default=9999.0)
            return (min_pg, min_y)
        raw_chunks.sort(key=_chunk_reading_order)

        # ── Assign sequential chunk_ids; normalize None → "NIL" ──────────────
        for chunk in raw_chunks:
            chunk['chunk_id'] = chunk_counter
            chunk_counter    += 1
            if chunk.get('clause_id') is None:
                chunk['clause_id'] = "NIL"
            if chunk.get('level') is None:
                chunk['level'] = 0

        # ── STEP 5: Parent ID assignment via clause hierarchy stack ───────────
        raw_chunks = self._assign_parent_ids(raw_chunks)
        self._log_toc_debug_snapshot("post-parent-assign", raw_chunks)

        # ── STEP 5b: Link vision_footnote chunks to their visual parent ───────
        raw_chunks = self._assign_vision_footnote_parents(raw_chunks)

        # ── STEP 6: Annex/Appendix context propagation ────────────────────────
        raw_chunks = self._detect_annex_appendix(raw_chunks)

        # Combine with any previously checkpointed chunks (resume case)
        all_chunks.extend(raw_chunks)

        # Attach page images to nearest chunk by Y-position (unchanged logic)
        if all_page_images and all_chunks:
            for img_entry in all_page_images:
                p = img_entry['page']
                page_chunks = [
                    c for c in all_chunks
                    if str(p) in
                       [x.strip() for x in c.get('source_page', '').split(',')]
                ]
                if page_chunks:
                    target = self._select_table_target_chunk(
                        page_chunks, p,
                        table_bbox=[0, img_entry['y0'], 0, img_entry['y1']]
                    )
                else:
                    target = all_chunks[-1]
                if target is not None:
                    target.setdefault('images', []).append(img_entry['image'])

        # Strip private helper keys before handing off to downstream consumers
        for c in all_chunks:
            c.pop('_first_line_rich_spans', None)
            c.pop('_second_line_rich_spans', None)
            c.pop('_block_left_x', None)
            c.pop('_block_label', None)

        return all_chunks, metrics

    def _parse_paddle_output(self, output, model_type):
        normalized_blocks = []
        if not output: return []

        if model_type == "v3":
            _item0 = output[0]
            paddle_data = (_item0.json if hasattr(_item0, 'json') else _item0).get('res', {})
            structure_blocks = paddle_data.get('parsing_res_list', [])
            layout_boxes = paddle_data.get('layout_det_res', {}).get('boxes', [])

            for s_block in structure_blocks:
                best_label = s_block.get('block_label', 'text')
                max_iou = 0.0
                s_bbox = s_block.get('block_bbox')
                if not s_bbox: continue
                for l_box in layout_boxes:
                    iou = self._calculate_iou(s_bbox, l_box.get('coordinate'))
                    if iou > max_iou:
                        max_iou = iou
                        best_label = l_box.get('label')
                if max_iou > 0.80:
                    s_block['block_label'] = best_label

            for i, blk in enumerate(structure_blocks):
                normalized_blocks.append({
                    'label': blk.get('block_label', 'text'),
                    'bbox': blk.get('block_bbox'),
                    'order': blk.get('block_order') if blk.get('block_order') is not None else i,
                    'ocr_text': (blk.get('text') or blk.get('content') or blk.get('raw_text')
                                 or blk.get('parsing_text') or blk.get('ocr_text') or blk.get('rec_text') or '')
                })

        elif model_type == "vl":
            res_list = list(output)
            if not res_list: return []
            page_res = res_list[0]
            data = page_res.json if hasattr(page_res, 'json') else page_res

            raw_blocks = []
            if 'res' in data and 'parsing_res_list' in data['res']:
                raw_blocks = data['res']['parsing_res_list']
            elif 'res' in data and 'layout_det_res' in data['res']:
                raw_blocks = data['res']['layout_det_res']['boxes']

            for i, blk in enumerate(raw_blocks):
                bbox = blk.get('block_bbox') or blk.get('coordinate') or blk.get('bbox')
                label = blk.get('block_label') or blk.get('label') or 'text'
                if bbox:
                    normalized_blocks.append({
                        'label': label,
                        'bbox': bbox,
                        'order': blk.get('block_order') if blk.get('block_order') is not None else i,
                        'ocr_text': (blk.get('text') or blk.get('content') or blk.get('raw_text')
                                 or blk.get('parsing_text') or blk.get('ocr_text') or blk.get('rec_text') or '')
                    })
        return normalized_blocks

    # Font names that use non-standard encoding and need the symbol map applied.
    _SYMBOL_FONT_KEYWORDS = frozenset([
        'symbol', 'math', 'stix', 'wingding', 'webding', 'zapf', 'dingbat',
        'mtextra', 'mathtype', 'euclid', 'cambria math', 'asana', 'xits',
    ])

    def _clean_text(self, text, font_name):
        if not text: return ""
        font_lower = font_name.lower()
        if any(kw in font_lower for kw in self._SYMBOL_FONT_KEYWORDS):
            mapping = self._get_symbol_map()
            text = "".join([mapping.get(c, c) for c in text])
        # Strip C0/C1 control characters that sneak in from bad font encodings,
        # but preserve tab (\x09), LF (\x0a), and CR (\x0d).
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Fix degrees Celsius formatting often broken in OCR
        text = re.sub(r'(\d+)\s*[\u03b8q0]\s*([CF])', r'\1°\2', text)
        # Fix tolerance symbols
        text = re.sub(r'\s+[\u03c1rp]\s+(?=\d)', r' ± ', text)
        text = unicodedata.normalize('NFKC', text)
        # ftfy fixes mojibake (text decoded with the wrong encoding, e.g. the
        # "Revision 2−DC2 Erratum" class of corruption from non-standard PDF fonts).
        text = ftfy.fix_text(text, normalization='NFKC')
        return text

    def _restore_missing_structural_prefix(self, fitz_lines, ocr_text):
        """
        Restore a dropped leading clause/list marker when Paddle's OCR text
        clearly contains it but the PyMuPDF text clipped from the same layout
        block does not.

        Typical recovery target:
          OCR  -> "1. Scope"
          fitz -> "Scope"
        Returns the original fitz_lines unless the match is high-confidence.
        """
        if not fitz_lines or not ocr_text:
            return fitz_lines

        fitz_lines = [str(ln).strip() for ln in fitz_lines if str(ln).strip()]
        if not fitz_lines:
            return fitz_lines

        fitz_first = fitz_lines[0]
        ocr_flat = re.sub(r'\s+', ' ', str(ocr_text or '').strip())
        if not ocr_flat:
            return fitz_lines

        # Only intervene for compact structural/list markers, not arbitrary
        # leading numbers such as years or document IDs.
        m = re.match(
            r'^\s*('
            r'(?:\d{1,3}(?:\.\d{1,3})*|[A-Z]\d+[A-Z]?|[ivxlcdm]{1,8}|[A-Za-z])'
            r'[\.)]?'
            r')\s+(.+?)\s*$',
            ocr_flat,
            re.IGNORECASE,
        )
        if not m:
            return fitz_lines

        marker = m.group(1).strip()
        ocr_title = m.group(2).strip()

        if re.match(r'^(?:\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$', marker):
            return fitz_lines
        if re.match(r'^(?:\d{1,2}:\d{2}(?::\d{2})?)$', marker):
            return fitz_lines
        if re.match(
            r'^\s*(?:'
            r'(?:\d{1,3}(?:\.\d{1,3})*|[A-Z]\d+[A-Z]?|[ivxlcdm]{1,8}|[A-Za-z])'
            r'[\.)]?'
            r')\b',
            fitz_first,
            re.IGNORECASE,
        ):
            return fitz_lines

        def _norm(s):
            return re.sub(r'[^a-z0-9]+', ' ', str(s or '').lower()).strip()

        fitz_first_norm = _norm(fitz_first)
        ocr_title_norm = _norm(ocr_title)
        if not fitz_first_norm or not ocr_title_norm:
            return fitz_lines

        if fitz_first_norm == ocr_title_norm or fitz_first_norm in ocr_title_norm or ocr_title_norm in fitz_first_norm:
            return [marker, *fitz_lines]

        return fitz_lines

    def _normalize_running_text(self, text, strip_variable_tokens=False):
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if strip_variable_tokens:
            text = re.sub(r'(?i)\bpage\s+\d+\s+of\s+\d+\b', 'page N of M', text)
            text = re.sub(r'(?i)\bpage\s*:\s*\d+\b', 'page:', text)
            text = re.sub(r'(?i)\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ' ', text)
            text = re.sub(r'(?i)\b\d{4}-\d{2}-\d{2}\b', ' ', text)
            text = re.sub(r'(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*/\d{4}\b', ' ', text)
            text = re.sub(r'(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', ' ', text)
            text = re.sub(r'(?i)\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', ' ', text)
            text = re.sub(r'(?i)\b\d{1,2}:\d{2}(?::\d{2})?\b', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text.casefold()

    # Unicode ranges for non-Latin scripts worth detecting.
    _NON_LATIN_RANGES = (
        (0x0900, 0x097F),   # Devanagari (Hindi, Sanskrit)
        (0x0600, 0x06FF),   # Arabic
        (0x0750, 0x077F),   # Arabic Supplement
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Chinese/Japanese/Korean)
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0xAC00, 0xD7AF),   # Hangul (Korean)
        (0x0400, 0x04FF),   # Cyrillic
        (0x0590, 0x05FF),   # Hebrew
        (0x0E00, 0x0E7F),   # Thai
    )

    def _detect_non_latin_script(self, text):
        """Returns True if text contains a meaningful amount of non-Latin characters."""
        if not text:
            return False
        hits = sum(
            1 for ch in text
            if any(lo <= ord(ch) <= hi for lo, hi in self._NON_LATIN_RANGES)
        )
        return hits > 3

    def _line_center_from_words(self, line_words):
        if not line_words:
            return None
        ys = []
        for word in line_words:
            bbox = word.get("bbox", [])
            if len(bbox) == 4:
                ys.append((bbox[1] + bbox[3]) / 2.0)
        if not ys:
            return None
        return sum(ys) / len(ys)

    def _normalize_edge_comparison_text(self, text):
        normalized = self._normalize_running_text(text, strip_variable_tokens=True)
        normalized = re.sub(r'(?i)\bpage:\b', 'page', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _footer_marker_positions(self, text):
        marker_pattern = re.compile(
            r'(?i)\b(title\s*:|country\s*:|original\s*:|page\s*:|regulation\s+no\.?|interregs|licensed\s+to|generated\s+on)\b'
        )
        return [match.start() for match in marker_pattern.finditer(text or "")]

    _STRONG_METADATA_RE = re.compile(
        r'(?i)\b(title\s*:|country\s*:|original\s*:|page\s*:|regulation\s+no\.?'
        r'|interregs|licensed\s+to|generated\s+on)\b'
    )

    def _has_strong_metadata_marker(self, text):
        """Return True if the text contains a known header/footer metadata marker."""
        return bool(self._STRONG_METADATA_RE.search(text or ""))

    def _matches_repeated_footer_signature(self, footer_text, repeated_footers):
        normalized_footer = self._normalize_edge_comparison_text(footer_text)
        if not normalized_footer:
            return False

        # Exact match: text was positively identified as a repeated footer during
        # detection — drop it without requiring metadata markers.
        if normalized_footer in repeated_footers:
            return True

        # Fuzzy / substring matching: require 2+ metadata markers to avoid
        # accidentally dropping real content near the page bottom.
        metadata_patterns = [
            r'\btitle\s*:',
            r'\bcountry\s*:',
            r'\boriginal\s*:',
            r'\bpage\b',
            r'\bregulation\s+no\b',
            r'\binterregs\b',
            r'\blicensed\s+to\b',
            r'\bgenerated\s+on\b',
        ]
        metadata_hits = sum(1 for pattern in metadata_patterns if re.search(pattern, normalized_footer, re.IGNORECASE))
        if metadata_hits < 2:
            return False

        for repeated in repeated_footers:
            if normalized_footer in repeated or repeated in normalized_footer:
                return True
        return False

    def _matches_repeated_header_signature(self, header_text, repeated_headers):
        normalized_header = self._normalize_edge_comparison_text(header_text)
        if not normalized_header:
            return False

        if normalized_header in repeated_headers:
            return True

        for repeated in repeated_headers:
            if normalized_header in repeated or repeated in normalized_header:
                return True
        return False

    def _strip_repeated_footer_suffix(self, line, repeated_footers):
        if not line or not repeated_footers:
            return line, None

        marker_positions = [pos for pos in self._footer_marker_positions(line) if pos > 0]
        if not marker_positions:
            return line, None

        for start in marker_positions:
            prefix = line[:start].rstrip()
            suffix = line[start:].strip()
            if not prefix or len(prefix.split()) < 4:
                continue
            if self._matches_repeated_footer_signature(suffix, repeated_footers):
                return prefix, suffix

        return line, None

    def _strip_known_footer_boilerplate_from_line(self, line):
        if not line:
            return line

        title_match = re.search(r'(?i)\btitle\s*:', line)
        if title_match:
            suffix = line[title_match.start():]
            if (
                re.search(r'(?i)\bregulation\s+no\b', suffix)
                or re.search(r'(?i)\bcountry\s*:', suffix)
                or re.search(r'(?i)\boriginal\s*:', suffix)
                or re.search(r'(?i)\bpage\s*:', suffix)
                or re.search(r'(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*/\d{4}\b', suffix)
            ):
                return line[:title_match.start()].rstrip()

        if re.match(r'(?i)^\s*country\s*:', line) and re.search(r'(?i)\boriginal\s*:', line):
            return ""
        if re.match(r'(?i)^\s*title\s*:', line) and re.search(r'(?i)\bregulation\s+no\b', line):
            return ""

        return line

    def _determine_text_type(self, found_type, block_label, content=""):
        """Derive a text_type tag from heading-detection result and PaddleOCR block label."""
        if found_type == 'toc_start_heading':
            return "toc"
        if found_type in (
            'alpha_paren_heading', 'numeric_paren_heading', 'simple_list_item',
            'alpha_bare_heading', 'roman_lower_paren_heading',
            'roman_lower_bare_heading', 'bullet_heading',
        ):
            return "list"
        heading_types = {
            'multi_level_heading', 'appendix_heading', 'chapter_heading',
            'content_start_heading', 'preamble_heading', 'article_heading',
            'roman_upper_heading', 'roman_upper_alpha_heading', 'section_general_heading',
            'cfr_part_heading', 'cfr_section_heading', 'fmvss_paragraph',
            'alpha_upper_heading', 'definition_heading', 'implicit_heading',
            'ece_regulation_heading',
        }
        if found_type in heading_types:
            return "heading"
        label = (block_label or "").lower()
        if any(x in label for x in ("equation", "formula", "display_formula")):
            return "equation"
        if "table" in label:
            return "table"
        if "figure_title" in label:
            return "figure_title"
        if "image_caption" in label or label == "caption":
            return "image_caption"
        return "text"

    def _refine_text_type(self, text_type, content, level=None):
        """Post-process chunks using content patterns (definition, note, unknown, subheading)."""
        t = (content or "").strip()
        if text_type == "heading":
            # Content-pattern overrides take priority over structural type
            if t and re.match(r'(?i)^\s*note\s*[:\-]', t):
                return "note"
            if t and re.search(r'[\u201c\u201d"][^"\u201c\u201d]{1,80}[\u201c\u201d"]\s+means\b', t):
                return "definition"
            # Level-based heading vs subheading (level 1 = top-level heading)
            if level is not None and level >= 2:
                return "subheading"
            return "heading"
        if text_type == "text":
            if not t:
                return "unknown"
            if re.match(r'(?i)^\s*note\s*[:\-]', t):
                return "note"
            if re.search(r'[\u201c\u201d"][^"\u201c\u201d]{1,80}[\u201c\u201d"]\s+means\b', t):
                return "definition"
            if re.match(r'^\s*[A-Z][A-Za-z\s\(\)\-]{1,80}\s+means\b', t):
                return "definition"
            return "text"
        return text_type

    def _sanitize_chunk_content_verbatim(self, text):
        if not text:
            return text

        table_marker_line_patterns = [
            re.compile(r'^\s*<Table_\d+_Pg\d+>\s*$', re.IGNORECASE),
            re.compile(r'^\s*\[\[TABLE_DETECTED_PAGE_\d+\]\]\s*$', re.IGNORECASE),
        ]

        cleaned_lines = []
        for raw_line in text.splitlines():
            cleaned_line = raw_line
            for pat in table_marker_line_patterns:
                cleaned_line = pat.sub("", cleaned_line)
            cleaned_line = self._strip_known_footer_boilerplate_from_line(cleaned_line).strip()
            if not cleaned_line:
                continue
            # Drop single-character OCR fragments that arise when a bbox clips
            # through a word — e.g. "i", "S", "y" extracted from a TOC or
            # paragraph block whose bbox edge cuts through surrounding text.
            if self._is_garbled_stamp_line(cleaned_line):
                continue
            cleaned_lines.append(cleaned_line)

        sanitized = "\n".join(cleaned_lines).strip()
        return sanitized

    def _merge_tables_html(self, existing_html, incoming_html, prefer_incoming=True):
        existing = [h for h in (existing_html or []) if h]
        incoming = [h for h in (incoming_html or []) if h]
        ordered = incoming + existing if prefer_incoming else existing + incoming

        seen = set()
        merged = []
        for html in ordered:
            if html in seen:
                continue
            seen.add(html)
            merged.append(html)
        return merged

    def _chunk_intersects_debug_pages(self, chunk, debug_pages=None):
        debug_pages = set(debug_pages or TOC_DEBUG_PAGES)
        if not debug_pages:
            return False

        page_keys = set()
        for pg in (chunk.get('block_bboxes', {}) or {}).keys():
            try:
                page_keys.add(int(pg))
            except Exception:
                pass
        if page_keys & debug_pages:
            return True

        for pg in str(chunk.get('source_page', '') or '').split(','):
            pg = pg.strip()
            if pg.isdigit() and int(pg) in debug_pages:
                return True
        return False

    def _log_toc_debug_snapshot(self, stage, chunks, debug_pages=None, limit=120):
        if not self.session_logger:
            return

        debug_pages = set(debug_pages or TOC_DEBUG_PAGES)
        if not debug_pages:
            return

        try:
            self.session_logger.info(
                f"[TOCDBG][{stage}] pages={sorted(debug_pages)} total_chunks={len(chunks)}"
            )
            count = 0
            for idx, chunk in enumerate(chunks):
                if not self._chunk_intersects_debug_pages(chunk, debug_pages):
                    continue
                count += 1
                pages = sorted(
                    int(pg) for pg in (chunk.get('block_bboxes', {}) or {}).keys()
                    if str(pg).isdigit()
                )
                bbox_summary = []
                for pg in sorted((chunk.get('block_bboxes', {}) or {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 9999):
                    boxes = (chunk.get('block_bboxes', {}) or {}).get(pg) or []
                    if boxes:
                        bbox_summary.append(f"p{pg}:{boxes[0]}")
                content = str(chunk.get('content_verbatim', '') or '')
                content = content.replace('\n', ' | ')
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 220:
                    content = content[:220] + "..."
                self.session_logger.info(
                    f"[TOCDBG][{stage}] idx={idx} chunk_id={chunk.get('chunk_id')} "
                    f"pages={pages or chunk.get('source_page', '')!r} "
                    f"label={chunk.get('_block_label', '')!r} text_type={chunk.get('text_type', '')!r} "
                    f"cid={chunk.get('clause_id')!r} title={chunk.get('title')!r} "
                    f"found_type={chunk.get('_found_type', '')!r} "
                    f"bbox={'; '.join(bbox_summary)} content={content!r}"
                )
                if count >= limit:
                    self.session_logger.info(
                        f"[TOCDBG][{stage}] truncated after {limit} matching chunks"
                    )
                    break
        except Exception as e:
            self.session_logger.warning(f"[TOCDBG][{stage}] logging failed: {e}")

    def _dedupe_tables_html(self, table_html_list):
        return self._merge_tables_html([], table_html_list, prefer_incoming=True)

    def _fix_cross_page_table_continuation(self, chunks):
        """
        Fix tables that are absorbed by edge chunks (header, footer, footnote).

        Problem:
          When a table is detected near a page header, footer, or footnote the
          spatial selector may assign it to that edge chunk instead of the
          surrounding content chunk.  This includes cross-page table splits
          where the continuation rows land on the next page's running header.

        Fix (applied to every edge chunk that carries tables_html):
          Option 1 — if the orphaned table has <th> column headers, scan back
            up to 30 chunks for a content chunk whose table has identical
            headers; merge the orphaned <tbody> rows into it.
          Default — if Option 1 finds no match (or the table has no <th>),
            append the whole table as a new tables_html entry on the nearest
            preceding non-edge content chunk.
          Last resort — if there is genuinely no content chunk above (e.g. the
            edge chunk is the very first thing on the first page), leave the
            table on the edge chunk so nothing is lost.
        """
        EDGE_TYPES = {'header', 'footer', 'footnote', 'vision_footnote'}

        def _get_th_headers(html):
            ths = re.findall(r'<th[^>]*>(.*?)</th>', html, re.DOTALL | re.IGNORECASE)
            return tuple(re.sub(r'<[^>]+>', '', th).strip() for th in ths)

        def _get_tbody_content(html):
            m = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        def _append_tbody_rows(base_html, extra_rows_html):
            if not extra_rows_html:
                return base_html
            return re.sub(
                r'(</tbody>)',
                extra_rows_html + r'\1',
                base_html,
                count=1,
                flags=re.IGNORECASE,
            )

        def _get_table_data(chunk):
            """Return the table_data list from the tables_html field (handles both old list and new dict format)."""
            tbl = chunk.get('tables_html')
            if isinstance(tbl, dict):
                return tbl.get('table_data', [])
            if isinstance(tbl, list):
                return tbl
            return []

        def _get_table_bbox(chunk):
            """Return the bbox dict from the tables_html field."""
            tbl = chunk.get('tables_html')
            if isinstance(tbl, dict):
                return tbl.get('bbox', {})
            return {}

        def _set_tables_html(chunk, table_data, bbox):
            """Write back the tables_html field in the new dict format."""
            chunk['tables_html'] = {'table_data': table_data, 'bbox': bbox}

        def _merge_bbox_dicts(a, b):
            """Merge two {page: [rects]} dicts together."""
            out = {k: list(v) for k, v in a.items()}
            for page, rects in b.items():
                out.setdefault(page, []).extend(rects)
            return out

        def _push_to_prev_chunk(orphan_html, orphan_bbox, chunks, from_index):
            """
            Find the nearest preceding non-edge content chunk and append
            orphan_html as a new tables_html entry on it.
            Returns True if successfully placed, False if no suitable chunk found.
            """
            for j in range(from_index - 1, max(from_index - 30, -1), -1):
                prev = chunks[j]
                if prev.get('text_type') in EDGE_TYPES:
                    continue
                prev_data = _get_table_data(prev)
                prev_bbox = _get_table_bbox(prev)
                prev_data.append(orphan_html)
                merged = _merge_bbox_dicts(prev_bbox, orphan_bbox or {})
                _set_tables_html(prev, prev_data, merged)
                return True
            return False

        for i, chunk in enumerate(chunks):
            # Process any edge chunk that has absorbed tables.
            if chunk.get('text_type') not in EDGE_TYPES:
                continue
            table_data = _get_table_data(chunk)
            if not table_data:
                continue

            chunk_bbox = _get_table_bbox(chunk)
            unmatched = []
            for idx, orphan_html in enumerate(table_data):
                orphan_headers = _get_th_headers(orphan_html)

                # Option 1: if the table has <th> headers, try to merge its
                # rows into an earlier chunk's table with matching columns.
                matched = False
                if orphan_headers:
                    for j in range(i - 1, max(i - 30, -1), -1):
                        prev = chunks[j]
                        prev_data = _get_table_data(prev)
                        for k, prev_html in enumerate(prev_data):
                            if _get_th_headers(prev_html) == orphan_headers:
                                extra_rows = _get_tbody_content(orphan_html)
                                prev_data[k] = _append_tbody_rows(prev_html, extra_rows)
                                _set_tables_html(prev, prev_data, _get_table_bbox(prev))
                                matched = True
                                break
                        if matched:
                            break

                # Default: no column-header match (or no <th> at all) —
                # append the whole table to the nearest preceding content chunk.
                if not matched:
                    if not _push_to_prev_chunk(orphan_html, chunk_bbox, chunks, i):
                        # Last resort: no content chunk found above; keep here.
                        unmatched.append(orphan_html)

            _set_tables_html(chunk, unmatched, chunk_bbox if unmatched else {})

        return chunks

    def _get_chunk_page_bounds(self, chunk, page_num):
        page_words = [
            w for w in chunk.get('content_words', [])
            if str(w.get('page', '')) == str(page_num)
            and isinstance(w.get('bbox'), list)
            and len(w.get('bbox')) >= 4
        ]
        if not page_words:
            return None

        y0 = min(float(w['bbox'][1]) for w in page_words)
        y1 = max(float(w['bbox'][3]) for w in page_words)
        return {"y0": y0, "y1": y1, "y_center": (y0 + y1) / 2.0}

    def _select_table_target_chunk(self, page_chunks, page_num, table_rect=None, table_bbox=None):
        if not page_chunks:
            return None
        if table_rect is None and table_bbox is None:
            return page_chunks[0]

        if table_rect is not None:
            table_top = float(table_rect.y0)
            table_center = float((table_rect.y0 + table_rect.y1) / 2.0)
        else:
            table_top = float(table_bbox[1])
            table_center = float((table_bbox[1] + table_bbox[3]) / 2.0)

        chunk_bounds = []
        for chunk in page_chunks:
            bounds = self._get_chunk_page_bounds(chunk, page_num)
            if bounds:
                chunk_bounds.append((chunk, bounds))
        if not chunk_bounds:
            # No word-level bounds found (e.g. heading-only chunks with no body
            # text yet). Return the most recent chunk by chunk_id as the best
            # positional proxy — it's more likely the heading just above the image.
            return max(page_chunks, key=lambda c: c.get('chunk_id', 0))

        preceding = [(chunk, bounds) for chunk, bounds in chunk_bounds if bounds['y1'] <= table_top]
        if preceding:
            return max(preceding, key=lambda x: x[1]['y1'])[0]

        overlapping = [
            (chunk, bounds) for chunk, bounds in chunk_bounds
            if bounds['y0'] <= table_top <= bounds['y1']
        ]
        if overlapping:
            return min(overlapping, key=lambda x: abs(x[1]['y_center'] - table_center))[0]

        return min(chunk_bounds, key=lambda x: abs(x[1]['y_center'] - table_center))[0]

    def _attach_ai_tables_to_chunks_by_page(self, chunks, page_num, table_html_list, rect=None):
        if not table_html_list:
            return

        page_chunks = []
        for chunk in chunks:
            pages = [p.strip() for p in chunk.get('source_page', '').split(',')]
            if str(page_num) in pages:
                page_chunks.append(chunk)
        if not page_chunks:
            return

        target_chunk = self._select_table_target_chunk(page_chunks, page_num, table_rect=rect)
        if target_chunk is None:
            return

        target_chunk['tables_html'] = self._dedupe_tables_html(table_html_list)

    def _should_preserve_edge_line(self, line, line_words, y_center, header_cutoff, footer_cutoff):
        if not line or y_center is None:
            return True

        if header_cutoff < y_center < footer_cutoff:
            return True

        clean_line = line.strip()
        if not clean_line:
            return True

        if re.match(r'^\s*(\d{1,3}(?:\.\d+)*|[A-Z]\d+(?:\.\d+)*)\.?\s+', clean_line):
            return True
        # Also protect bare clause numbers with no text on the same line
        # (e.g. "1.", "1.1.", "2.3.") — PaddleOCR often splits the number
        # onto its own OCR line.  Without this, a section number that appears
        # near the top of many pages can be mis-detected as a repeated running
        # header and stripped from body content.
        if re.match(r'^\s*\d{1,3}(?:\.\d+)*\.?\s*$', clean_line):
            return True
        if re.match(r'(?i)^\s*(annex|appendix)\s+[A-Z0-9]+', clean_line):
            return True
        if re.search(r'\.{3,}\s*\d+\s*$', clean_line):
            return True
        if len(clean_line.split()) >= 12 and re.search(r'[.;:]$', clean_line):
            return True
        if any(word.get("text", "").strip().isdigit() for word in line_words[:1]):
            return True

        return False

    def _is_safe_repeated_edge_line(self, line, line_words, header_cutoff, footer_cutoff, repeated_headers, repeated_footers):
        normalized_line = self._normalize_edge_comparison_text(line)

        y_center = self._line_center_from_words(line_words)
        near_top = y_center is not None and y_center <= header_cutoff
        near_bottom = y_center is not None and y_center >= footer_cutoff

        if not near_top and not near_bottom:
            return False
        if self._should_preserve_edge_line(line, line_words, y_center, header_cutoff, footer_cutoff):
            return False

        # A line near an edge whose normalized form is empty consists entirely of
        # variable tokens (date stamps like "Mar/2025", standalone page numbers,
        # version strings) — always treat it as footer/header noise.
        if not normalized_line:
            return True

        if near_top and self._matches_repeated_header_signature(line, repeated_headers):
            return True
        if near_bottom and self._matches_repeated_footer_signature(line, repeated_footers):
            return True
        return False

    def _block_is_in_header_footer_zone(self, block, scale_x, scale_y,
                                         header_cutoff, footer_cutoff,
                                         repeated_headers, repeated_footers, page):
        """Return 'header', 'footer', or None.
        A non-None result means the block sits entirely within a header or footer
        zone and its text matches a repeated signature or contains a strong metadata
        marker. This catches layout-model blocks labeled 'text' that are actually
        page chrome — they will be tagged rather than dropped."""
        bbox = block['bbox']
        pdf_y1 = bbox[1] * scale_y
        pdf_y2 = bbox[3] * scale_y

        in_header_zone = pdf_y2 <= header_cutoff   # block top-edge entirely above cutoff
        in_footer_zone = pdf_y1 >= footer_cutoff   # block bottom-edge entirely below cutoff
        if not in_header_zone and not in_footer_zone:
            return None

        pdf_rect = fitz.Rect(
            bbox[0] * scale_x, pdf_y1,
            bbox[2] * scale_x, pdf_y2,
        )
        block_text = page.get_text(clip=pdf_rect).strip()
        if not block_text:
            return None

        if in_header_zone and self._matches_repeated_header_signature(block_text, repeated_headers):
            return "header"
        if in_footer_zone and self._matches_repeated_footer_signature(block_text, repeated_footers):
            return "footer"
        # Strong semantic marker overrides repetition requirement
        if self._has_strong_metadata_marker(block_text):
            return "header" if in_header_zone else "footer"
        return None

    def _synthesize_summary_for_category(self, category_name, summaries, quotes, model="gpt-4o"):
            if not self._has_ai() or not summaries:
                return "No summary available.", "No quote available."

            # 1. Synthesize the summary
            summary_input = "\n".join(f"- {s}" for s in summaries)
            summary_prompt = f"Synthesize these individual points about '{category_name}' into one cohesive paragraph that covers the key aspects. Be concise."

            try:
                final_summary = self._call_llm(
                    model=model,
                    system_prompt="You are a technical summarizer.",
                    user_prompt=f"POINTS:\n{summary_input}\n\nSYNTHESIZED SUMMARY:",
                    temperature=0.1,
                    max_tokens=250
                )
            except Exception:
                final_summary = "Could not generate summary."

            # 2. Select the best quote
            quote_input = "\n".join(f"- \"{q}\"" for q in quotes if q)
            quote_prompt = f"From the following list of quotes for '{category_name}', select the one that best represents the primary, most binding requirement. Return only that single quote, verbatim, without any extra text or quotation marks."

            try:
                final_quote = self._call_llm(
                    model=model,
                    system_prompt="You are a compliance expert who selects the most critical verbatim text.",
                    user_prompt=f"QUOTES:\n{quote_input}\n\nBEST QUOTE:",
                    temperature=0.0,
                    max_tokens=250
                )
                # Clean up potential AI artifacts like surrounding quotes
                if final_quote.startswith('"') and final_quote.endswith('"'):
                    final_quote = final_quote[1:-1]

            except Exception:
                final_quote = "Could not select quote."

            return final_summary, final_quote

    def detect_header_footer_zones(self, doc, sample_pages=20, top_margin_percent=12, bottom_margin_percent=10):
        page_height = doc[0].rect.height
        top = page_height * (top_margin_percent / 100.0)
        bottom = page_height * (1 - bottom_margin_percent / 100.0)
        header_cands = defaultdict(set)
        footer_cands = defaultdict(set)
        # Lines with strong metadata markers bypass the repetition threshold
        strong_header_cands = set()
        strong_footer_cands = set()

        limit = min(len(doc), sample_pages)
        for i in range(limit):
            page = doc[i]
            page_num = i + 1
            page_dict = page.get_text("dict")
            seen_headers = set()
            seen_footers = set()
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    line_text_parts = []
                    line_words = []
                    for span in line.get("spans", []):
                        txt = self._clean_text(span.get("text", ""), span.get("font", ""))
                        if txt.strip():
                            line_text_parts.append(txt)
                            line_words.append({"bbox": [round(x) for x in span.get("bbox", [])], "text": txt})
                    full_line = " ".join(line_text_parts).strip()
                    if not full_line:
                        continue
                    y_center = self._line_center_from_words(line_words)
                    if y_center is None:
                        continue
                    comparison_key = self._normalize_edge_comparison_text(full_line)
                    if not comparison_key:
                        continue
                    if self._should_preserve_edge_line(full_line, line_words, y_center, top, bottom):
                        continue
                    if y_center <= top and comparison_key not in seen_headers:
                        header_cands[comparison_key].add(page_num)
                        seen_headers.add(comparison_key)
                        if self._has_strong_metadata_marker(full_line):
                            strong_header_cands.add(comparison_key)
                    elif y_center >= bottom and comparison_key not in seen_footers:
                        footer_cands[comparison_key].add(page_num)
                        seen_footers.add(comparison_key)
                        if self._has_strong_metadata_marker(full_line):
                            strong_footer_cands.add(comparison_key)

        thresh = max(2, int(np.ceil(limit * 0.15)))
        headers = {k for k, pages in header_cands.items() if len(pages) >= thresh}
        footers = {k for k, pages in footer_cands.items() if len(pages) >= thresh}
        # Strong semantic markers: include even if they appear on only 1 page
        headers |= strong_header_cands
        footers |= strong_footer_cands
        return top, bottom, headers, footers

    def _sort_blocks_in_reading_order(self, blocks, page_width):
            """
            Sorts blocks to handle mixed layouts (1-column headers + 3-column text).
            Logic:
            1. Identify 'Anchors' (Wide blocks > 60% page width) like headers/footers.
            2. Use Anchors to split the page into vertical 'Segments'.
            3. Inside a Segment:
            - Group blocks into 'Columns' based on X-coordinates.
            - Sort Columns Left-to-Right.
            - Sort Blocks inside Columns Top-to-Bottom.
            """
            # Filter out tiny/empty blocks to reduce noise
            blocks = [b for b in blocks if b['bbox'][2] - b['bbox'][0] > 1 and b['bbox'][3] - b['bbox'][1] > 1]
            if not blocks: return []

            # Threshold: Blocks wider than 60% of page are considered 'Anchors' (Headers/Footers/Wide Tables)
            wide_thresh = page_width * 0.60

            # 1. Initial sort by Y to find vertical sequence
            blocks.sort(key=lambda b: b['bbox'][1])

            segments = []
            current_segment_blocks = []

            for b in blocks:
                width = b['bbox'][2] - b['bbox'][0]
                is_wide = width > wide_thresh

                if is_wide:
                    # If we have accumulated content blocks, push them as a segment first
                    if current_segment_blocks:
                        segments.append({'type': 'content', 'blocks': current_segment_blocks})
                        current_segment_blocks = []
                    # Push the wide anchor as its own segment
                    segments.append({'type': 'anchor', 'blocks': [b]})
                else:
                    current_segment_blocks.append(b)

            # Flush remaining content
            if current_segment_blocks:
                segments.append({'type': 'content', 'blocks': current_segment_blocks})

            # 2. Sort inside segments
            final_order = []
            for seg in segments:
                if seg['type'] == 'anchor':
                    # Anchors are already sorted by Y
                    final_order.extend(seg['blocks'])
                else:
                    # Content segment: Cluster into Columns
                    # Sort by Left Edge (X0) to identify columns
                    seg_blocks = seg['blocks']
                    seg_blocks.sort(key=lambda b: b['bbox'][0])

                    columns = []
                    if seg_blocks:
                        current_col = [seg_blocks[0]]
                        current_max_x = seg_blocks[0]['bbox'][2]

                        for b in seg_blocks[1:]:
                            # Check if this block starts significantly after the previous column ends
                            # Buffer of 10px handles minor layout drift
                            if b['bbox'][0] > current_max_x - 10:
                                # New column found
                                columns.append(current_col)
                                current_col = [b]
                                current_max_x = b['bbox'][2]
                            else:
                                # Belongs to current column (handling indentation)
                                current_col.append(b)
                                current_max_x = max(current_max_x, b['bbox'][2])
                        columns.append(current_col)

                    # Sort inside each column by Y (Top-to-Bottom)
                    for col in columns:
                        col.sort(key=lambda b: b['bbox'][1])
                        final_order.extend(col)

            return final_order
    def _is_span_bold(self, span):
        # Check flags (bit 4 is usually bold)
        if span["flags"] & 2 ** 4:
            return True

        # Check font name string for keywords
        font_name = span["font"].lower()
        if any(x in font_name for x in ["bold", "black", "heavy", "demi"]):
            return True

        return False


    def extract_content_with_stats(self, doc, num_pages_to_keep, paddle_model, model_type="vl"):
        logical_lines = []
        metrics = {
            "total_words": 0, "kept_words": 0, "dropped_words": 0,
            "debug_images": {}, "detected_tables": defaultdict(list),
            "removed_header_lines": 0, "removed_footer_lines": 0,
            "removed_edge_line_samples": []
        }

        limit = min(len(doc), num_pages_to_keep)

        for p_idx in range(limit):
            page = doc[p_idx]
            p_num = p_idx + 1

            # 1. Visual Prep
            zoom = 1.5  # VL model resizes internally; 2x gave no quality gain but doubled pixels
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            del pix, img_data  # free raw pixel buffer; img_bgr is an independent copy

            # 2. Paddle Prediction
            if model_type == "vl":
                layout_results = paddle_model.predict(img_bgr, use_queues=True, max_pixels=1003520)
            else:
                layout_results = paddle_model.predict(img_bgr)
            raw_blocks = self._parse_paddle_output(layout_results, model_type)

            # 3. Filter & Sort
            # Only strip bare page-number noise; all layout regions are now kept.
            excluded_labels = ['number', 'page_number']
            valid_blocks = [b for b in raw_blocks if b['label'].lower() not in excluded_labels]
            self._assign_synthetic_order(valid_blocks)
            valid_blocks.sort(key=self._get_sorting_key)

            # 4. Debug Visualization
            viz_image = img_bgr.copy()
            scale_x = page.rect.width / img_bgr.shape[1]
            scale_y = page.rect.height / img_bgr.shape[0]

            for idx, block in enumerate(raw_blocks):
                label = block['label'].lower()
                bbox = block['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # Draw Box
                is_excl = label in ['header', 'footer']
                color = (0, 0, 255) if 'table' in label else ((128, 128, 128) if is_excl else (0, 255, 0))
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                _order_num = block.get('order') if block.get('order') is not None else idx + 1
                cv2.putText(viz_image, f"{_order_num} {label.upper()}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 5. Text Extraction
                if not is_excl:
                    pdf_rect = fitz.Rect(bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y)

                    if 'table' in label:
                        # Register table for AI extraction later
                        metrics['detected_tables'][p_num].append(pdf_rect)
                        # Add placeholder to stream
                        logical_lines.append({
                            "text": f"[[TABLE_DETECTED_PAGE_{p_num}]]",
                            "words": [{"page": p_num, "text": "TABLE", "bbox": list(pdf_rect)}],
                            "page": p_num,
                            "rich_spans": []
                        })
                    else:
                        # Extract text with rich properties inside the block
                        block_dict = page.get_text("dict", clip=pdf_rect)
                        for b in block_dict.get("blocks", []):
                            if "lines" not in b: continue
                            for l in b["lines"]:
                                line_text_parts = []
                                line_words = []
                                line_rich = []

                                for s in l["spans"]:
                                    txt = self._clean_text(s["text"], s["font"])
                                    if txt.strip():
                                        line_text_parts.append(txt)
                                        is_bold = self._is_span_bold(s)
                                        line_rich.append({"text": txt, "is_bold": is_bold})
                                        line_words.append({
                                            "page": p_num, "text": txt, "bbox": [round(x) for x in s["bbox"]]
                                        })

                                full_line = " ".join(line_text_parts).strip()
                                if full_line:
                                    logical_lines.append({
                                        "text": full_line,
                                        "words": line_words,
                                        "page": p_num,
                                        "rich_spans": line_rich
                                    })
                                    metrics['kept_words'] += len(full_line.split())

            # Encode Debug Image for UI (cap at 50 pages to limit RAM)
            success, buffer = cv2.imencode(".jpg", viz_image)
            if success and len(metrics['debug_images']) < 50:
                metrics['debug_images'][p_num] = buffer.tobytes()

        return logical_lines, metrics

    def _extract_title_from_line(self, rich_spans, clause_id):
        if not rich_spans:
            return "NIL"

        title_parts = []
        capturing = False
        prefix_consumed = False

        # Build the set of textual forms the clause_id prefix can take so we
        # can skip that span rather than aborting when we see it non-bold.
        # E.g. clause_id="1"  ->  {"(1)", "1.", "1)", "1"}
        prefix_set = set()
        if clause_id and clause_id != "NIL":
            c = str(clause_id).strip().strip('.')
            prefix_set = {f"({c})", f"{c}.", f"{c})", c}

        for i, s in enumerate(rich_spans):
            text_clean = s['text'].strip()
            if not text_clean:
                continue

            is_bold = s['is_bold']
            is_quote = text_clean in ['"', "'", '“', '”']

            if is_bold:
                capturing = True
                title_parts.append(text_clean)
            elif is_quote and capturing:
                pass
            elif is_quote and not capturing:
                pass
            else:
                if capturing:
                    break
                elif not prefix_consumed and text_clean in prefix_set:
                    # Skip the structural prefix (e.g. “(1)”, “1.”) before
                    # looking for the inline bold term that follows it.
                    prefix_consumed = True
                    continue
                else:
                    return "NIL"

        if not title_parts:
            # Fallback: if no bold text found, use the whole line as the title
            # (covers PDFs that don't embed bold font flags in headings).
            full_text = " ".join([s['text'] for s in rich_spans]).strip()
            # Never return a bare structural prefix as the title.
            if full_text in prefix_set or not full_text:
                return "NIL"
            if len(full_text) < 100:
                raw_header = full_text
            else:
                return "NIL"
        else:
            raw_header = " ".join(title_parts).strip()

        # Remove the ID from the Title string if present
        if clause_id and clause_id != "NIL":
            clean_title = self._strip_heading_prefix(raw_header, clause_id)
        else:
            clean_title = raw_header

        clean_title = clean_title.strip('"\'\u201c\u201d')
        clean_title = clean_title.rstrip('.')

        if not clean_title:
            return "NIL"

        return clean_title

    def _normalize_heading_clause_id(self, raw_id, line, heading_type=None):
        raw_id = str(raw_id or "").strip().strip(".")
        line = str(line or "")
        if not raw_id or heading_type != "multi_level_heading":
            return raw_id or "NIL"

        # OCR sometimes inserts spaces around separators or between heading
        # number segments, e.g. "2 1." or "2 . 1.". Recover the intended
        # hierarchical clause ID from the visible line prefix.
        m = re.match(
            r'^\s*[\"\'“”]?((?:\d{1,3}\s*(?:\.\s*|\s+)){1,4}\d{1,3}|\d{1,3}(?:\.\d{1,3}){0,4})\s*\.?',
            line,
        )
        if not m:
            return raw_id

        candidate = m.group(1)

        # Before promoting spaces to dots, check the line suffix after the
        # matched candidate for OCR junk (e.g. "3 51/i" — the "/i" is noise).
        # If anything non-whitespace follows the candidate match, bail out.
        suffix = line[m.end():]
        if re.search(r'\S', suffix):
            return raw_id

        candidate = re.sub(r"\s*\.\s*", ".", candidate)

        # Only convert a space to a dot when both surrounding segments are
        # plausible sub-clause numbers (≤ 2 digits each). A segment like "51"
        # after a single-digit prefix is almost certainly OCR noise, not a real
        # clause like "3.51" that appeared nowhere in the document text.
        def _safe_space_to_dot(match_obj):
            before = match_obj.string[:match_obj.start()].split(".")[-1].strip()
            after  = match_obj.string[match_obj.end():].split(".")[0].strip()
            if len(before) <= 2 and len(after) <= 2:
                return "."
            return match_obj.group(0)

        candidate = re.sub(r"(?<=\d)\s+(?=\d)", _safe_space_to_dot, candidate)
        candidate = re.sub(r"\.+", ".", candidate).strip(".")

        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){0,4}", candidate):
            return candidate
        return raw_id

    def _strip_heading_prefix(self, line, clause_id):
        line = str(line or "")
        clause_id = str(clause_id or "").strip().strip(".")
        if not clause_id or clause_id == "NIL":
            return line.strip()

        parts = [re.escape(part) for part in clause_id.split(".") if part]
        if not parts:
            return line.strip()

        prefix_pattern = r"^\s*[\"'“”]?" + r"\s*(?:\.\s*|\s+)".join(parts) + r"\s*\.?\s*"
        return re.sub(prefix_pattern, "", line, count=1).strip()

    def get_compliance_patterns(self):
        return {
            'toc_start_heading': r'(?i)^\s*(?:\d+\.?\s+)?(TABLE\s+OF\s+CONTENTS|CONTENTS)\s*$',
            'preamble_heading': r'^\s*((?:AGENCY|ACTION|SUMMARY|DATES|ADDRESSES|SUPPLEMENTARY INFORMATION))\s*:?\s*(.*)',
            'chapter_heading': r'^\s*(?i:CHAPTER)\s+([A-Z0-9\-\.\s]+).*',
            'section_general_heading': r'^\s*(?i:SECTION)\s+([A-Z0-9\-\.\s]+).*',
            'cfr_part_heading': r'^\s*((?:49\s+CFR\s+)?PART\s+\d+).*',
            'cfr_section_heading': r'^\s*(§\s*\d+(?:\.\d+)?(?:[–\-—]\d{2,4})?)\s+(.*)',
            'fmvss_paragraph': r'^\s*(S\d+(\.\d+)*)\.?\s+(.*)',
            'article_heading': r'^\s*((?i:ARTICLE\s+\d+))\b\.?\s*(?:[A-Z0-9"“\'].*)?$',
            'appendix_heading': r'(?i)^\s*(APPENDIX|ANNEX)\s+([A-Z0-9]+)\s*(.*)',
            'roman_upper_heading': r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\.(?:\s+(?![A-Z0-9]\.)(.*)|$)',
            'roman_upper_alpha_heading': r'^\s*((?:XIV|XIII|XII|XI|IX|VIII|VII|VI|IV|III|II|XV|I|X|V)[a-z])\.(?:\s+(.*)|$)',
            'multi_level_heading': r"^\s*(?!49\s+CFR)[“”\’’’]?(\d{1,3}(?!\d)(\.\d{1,3}){0,9})(?!\d)\.?\s*(?:[A-Z0-9””\’’’].*)?$",
            'numeric_paren_heading': r'^\s*\((\d+)\)\s*(?![\d]{3}[-\s–\u2013\u2014])(.*)',
            'alpha_paren_heading': r'^\s*\(([a-z])\)\s*(.*)',
            # Lowercase roman numerals: (i), (ii), ... (xv) with parentheses
            'roman_lower_paren_heading': r'^\s*\((i{1,3}|i?v|vi{0,3}|ix|xi{0,3}|xi?v|xv)\)\s*(.*)',
            # Bare alpha without leading paren: “a) text” or “A) text”
            'alpha_bare_heading': r'^\s*([A-Za-z])\)\s*(.*)',
            # Bare lowercase roman without paren: “i) text”, “ii) text”
            'roman_lower_bare_heading': r'^\s*(i{1,3}|i?v|vi{0,3}|ix|xi{0,3}|xi?v|xv)\)\s*(.*)',
            # Bullet points: “• text”, “– text”, “- text”, “* text”
            'bullet_heading': r'^\s*([•\-–—\*])\s*(.*)',
            # --- Issue 2 fix: ECE / UN Regulation number headings (e.g. “ECE 14-08”,
            # “Regulation No. 14-08”, “UN R14”, “UNECE R14-08”) ---
            'ece_regulation_heading': r'^\s*((?:ECE|UNECE)\s+\d+[-\.]\d+|Regulation\s+No\.?\s*\d+[-\.]\d+|UN\s+R\d+(?:[-\.]\d+)?)\b',
            # CFR-style bare clause reference: "86.135–90" or "86.141" alone on its line
            # (number on one line, title on the next line)
            'cfr_clause_year_heading': r'^\s*(\d{1,3}\.\d{1,3}(?:\.\d+)*(?:[–\-—]\d{2,4})?)\s*$',
        }

    def _extract_candidates_from_text(self, text):
        candidates = []

        # 1. Standard Documents (Annex, Appendix, etc.)
        label_pattern = re.compile(
            r'(?i)\b(Annex(?:es)?|Append(?:ix|ices)|Tab(?:le|les)|Fig(?:ure|ures)|Sec(?:tion|tions)?|Art(?:icle|icles)?|Part|Clause|Diag(?:ram)?|Sheet)\s+'
            r'((?:(?!(?:Annex|Append|Tab|Fig|Sec|Art|Part|Clause|Diag|Sheet))[A-Z0-9\.\-]+(?:-[A-Z0-9\.\-]+)?(?:,\s*|,\s*and\s*|\s+and\s+)?)+)',
            re.IGNORECASE
        )
        for match in label_pattern.finditer(text):
            raw_type = match.group(1).lower()
            content_blob = match.group(2)

            # Normalize prefix
            prefix = raw_type.capitalize()
            if raw_type.startswith('annex'): prefix = "Annex"
            elif raw_type.startswith('append'): prefix = "Appendix"
            elif raw_type.startswith('tab'): prefix = "Table"
            elif raw_type.startswith('fig'): prefix = "Figure"
            elif raw_type.startswith('sec'): prefix = "Section"
            elif raw_type.startswith('art'): prefix = "Article"
            elif raw_type.startswith('part'): prefix = "Part"
            elif raw_type.startswith('clause'): prefix = "Clause"
            elif raw_type.startswith('diag'): prefix = "Diagram"
            elif raw_type.startswith('sheet'): prefix = "Sheet"

            # Split only on explicit list separators; the previous pattern could
            # match the empty string and break headings into single characters.
            items = re.split(r'\s*,\s*|\s+and\s+', content_blob)
            for item in items:
                clean_item = item.strip().strip('.')
                if not clean_item: continue
                if len(clean_item) == 1 and clean_item.islower(): continue
                if not self._is_compact_reference_token(clean_item): continue
                candidates.append(f"{prefix} {clean_item}")

        # 1b. Range References (e.g. "Articles 28 to 33", "Clauses 5.1 to 5.3")
        range_pattern = re.compile(
            r'(?i)\b(Annex(?:es)?|Append(?:ix|ices)|Tab(?:le|les)?|Fig(?:ure|ures)?|Sec(?:tion|tions)?|Art(?:icle|icles)?|Part|Clauses?|Diag(?:ram)?|Sheet)\s+'
            r'(\d+(?:\.\d+)*)\s+to\s+(\d+(?:\.\d+)*)'
        )
        for match in range_pattern.finditer(text):
            raw_type = match.group(1).lower()
            start_val = match.group(2)
            end_val   = match.group(3)
            r_prefix = raw_type.capitalize()
            if raw_type.startswith('annex'):   r_prefix = "Annex"
            elif raw_type.startswith('append'): r_prefix = "Appendix"
            elif raw_type.startswith('tab'):    r_prefix = "Table"
            elif raw_type.startswith('fig'):    r_prefix = "Figure"
            elif raw_type.startswith('sec'):    r_prefix = "Section"
            elif raw_type.startswith('art'):    r_prefix = "Article"
            elif raw_type.startswith('part'):   r_prefix = "Part"
            elif raw_type.startswith('clause'): r_prefix = "Clause"
            elif raw_type.startswith('diag'):   r_prefix = "Diagram"
            elif raw_type.startswith('sheet'):  r_prefix = "Sheet"
            try:
                s, e = int(start_val), int(end_val)
                if s < e <= s + 50:  # sanity cap: never expand more than 50 items
                    for i in range(s, e + 1):
                        candidates.append(f"{r_prefix} {i}")
            except ValueError:
                # dotted ranges — just add both endpoints
                candidates.append(f"{r_prefix} {start_val.strip('.')}")
                candidates.append(f"{r_prefix} {end_val.strip('.')}")

        # 2. Strict Numeric References (e.g. "3.1", "5.2.1")
        dotted_pattern = re.compile(r'\b(\d+(?:\.\d+)+)\b')
        ignore = {'mg', 'kg', 'mm', 'cm', 'm', 'g', 'hz', 'v', 'cfr', 'usc', 'iso', 'astm'}
        for match in dotted_pattern.finditer(text):
            val = match.group(1)
            next_chunk = text[match.end():match.end()+10].strip().lower()
            first_word = next_chunk.split()[0].strip('.,;:)') if next_chunk else ""
            if first_word in ignore: continue
            candidates.append(val)

        # 3. Generic Paragraph References (e.g. "paragraph 2.3", "paragraphs 2.3")
        generic_pattern = re.compile(r'(?i)\b(?:paragraphs?|items?|points?|para)\s+([A-Z0-9\.]+)\b')
        for match in generic_pattern.finditer(text):
            candidates.append(match.group(1).strip('.'))

        # --- NEW: US FMVSS References (e.g. "S5.1", "S14.2") ---
        # Look for Capital S followed immediately by a number
        fmvss_pattern = re.compile(r'\b(S\d+(?:\.\d+)*)\b')
        for match in fmvss_pattern.finditer(text):
            candidates.append(match.group(1).strip('.'))

        # --- NEW: CFR Section References (e.g. "§ 571.111") ---
        cfr_pattern = re.compile(r'(?:§|49\s+CFR)\s*(\d+(?:\.\d+)*)')
        for match in cfr_pattern.finditer(text):
            # Normalizing to just the number often matches better with ID logic
            # Or keep the symbol if your IDs use it.
            # Given our parsing logic extracted "§ 571.111", we should capture full string.
            full_ref = f"§ {match.group(1)}"
            candidates.append(full_ref)

        return list(set(candidates))

    def _extract_external_standard_refs(self, text):
        """
        Extract references to known external standards bodies.
        Patterns are intentionally tight: only well-known named prefixes
        followed by a numeric/alphanumeric identifier qualify.
        Free-form prose never passes through.
        """
        results = []
        patterns = [
            # ISO / IEC / IEEE (including combined, e.g. ISO/IEC 27001, ISO/IEC/IEEE 12207)
            r'\bISO(?:/IEC)?(?:/IEEE)?\s+\d[\w\-:\.\s]*?(?=\s{2,}|\s*[,;)]|$)',
            r'\bIEC\s+\d[\w\-:\.]*',
            r'\bIEEE\s+\d[\w\-:\.]*',
            # SAE (J-numbers, AS/AMS/ARP/AIR series, e.g. SAE J3068, SAE AS6081)
            r'\bSAE\s+(?:J|AS|AMS|ARP|AIR)\s*\d[\w\-]*',
            # FMVSS by name (e.g. FMVSS 126, FMVSS No. 208)
            r'\bFMVSS\s+(?:No\.?\s*)?\d+',
            # CFR (e.g. 49 CFR Part 571, 49 CFR 571.111)
            r'\b\d+\s+CFR\s+(?:(?:Part|§)\s*)?\d+(?:\.\d+)*',
            # Bare section symbol (e.g. § 571.111) — only when followed by dotted number
            r'§\s*\d+(?:\.\d+)+',
            # UL (Underwriters Laboratories, e.g. UL 94, UL 2054)
            r'\bUL\s+\d[\w\-]*',
            # ASTM (e.g. ASTM D638, ASTM B117)
            r'\bASTM\s+[A-Z]\d+[\w\-]*',
            # ANSI (e.g. ANSI Z87.1, ANSI/ISA 18.2)
            r'\bANSI(?:/[A-Z]+)?\s+[A-Z]?\d[\w\.\-]*(?:\s*[\:\-]\s*\d{4})?',
            # DIN / EN (e.g. DIN 72552, EN 13849, DIN EN ISO 9001)
            r'\bDIN\s+(?:EN\s+)?(?:ISO\s+)?\d[\w\-]*',
            r'\bEN\s+(?:ISO\s+)?\d[\w\-:\.]*',
            # MIL-STD (e.g. MIL-STD-1553, MIL-STD 810)
            r'\bMIL[\s\-–]+STD[\s\-–]*\d[\w\-]*',
            # NFPA (e.g. NFPA 70, NFPA 2112)
            r'\bNFPA\s+\d[\w\-]*',
            # AISI / AWS / ASME (e.g. ASME B31.3, AWS D1.1)
            r'\bASME\s+[A-Z]\d+[\w\.\-]*',
            r'\bAWS\s+[A-Z]\d+[\w\.\-]*',
            r'\bAISI\s+\d[\w\-]*',
            # GB / GB/T (Chinese national standards, e.g. GB/T 18384)
            r'\bGB(?:/T)?\s+\d[\w\-]*',
            # JASO (Japanese automotive, e.g. JASO M346)
            r'\bJASO\s+[A-Z]\d+[\w\-]*',
            # UNECE / UN ECE Regulations (e.g. UN ECE R100, UNECE Regulation No. 13)
            r'\bUN\s*[-–]?\s*ECE\s+(?:R|Regulation\s+(?:No\.?\s*)?)?\d+',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                ref = re.sub(r'\s+', ' ', m.group(0).strip().rstrip('.,;:'))
                if ref:
                    results.append(ref)
        return sorted(set(results))

    def _is_compact_reference_token(self, token):
        token = (token or "").strip().strip(".").strip("()[]{}")
        if not token or re.search(r"\s", token):
            return False
        if not re.fullmatch(r"[A-Z0-9]+(?:[.\-][A-Z0-9]+)*", token, re.IGNORECASE):
            return False

        # Reject long alphabetic words like heading titles ("CATEGORIZATION")
        # while still allowing short lettered annexes and roman numerals.
        if token.isalpha():
            upper = token.upper()
            if re.fullmatch(r"[IVXLCDM]+", upper):
                return True
            return len(upper) <= 4

        return True

    def _resolve_references(self, chunks):
        valid_ids = {c.get('clause_id', '').strip().lower() for c in chunks if c.get('clause_id') != "NIL"}
        for chunk in chunks:
            text = chunk.get('content_verbatim', '')
            cid = chunk.get('clause_id', '').strip().lower()
            # Chunks with dotted fill lines are TOC/form entries — no meaningful references
            if re.search(r'\.{10,}', text):
                chunk['references'] = []
                chunk['external_references'] = self._extract_external_standard_refs(text)
                continue
            raw = self._extract_candidates_from_text(text)
            confirmed = []
            structural_prefixes = ('annex', 'appendix', 'part')
            visual_prefixes = ('table', 'figure', 'fig', 'chart', 'diagram', 'sheet')
            if chunk.get('clause_id') == "TOC":
                for c in raw:
                    # In TOC, we primarily care about high-level structure
                    cl = c.lower()
                    if cl.startswith(structural_prefixes) and cl in valid_ids:
                        confirmed.append(c)
            else:
                for c in raw:
                    cl = c.strip().lower()
                    is_structural = cl.startswith(structural_prefixes)
                    is_vis = cl.startswith(visual_prefixes)

                    is_valid = cl in valid_ids

                    # Structural references must exist in the extracted document
                    # structure. Visual references can still pass through when
                    # they look like compact identifiers (e.g. Table 2, Figure A).
                    if is_structural:
                        keep = is_valid
                    elif is_vis:
                        suffix = c.split(" ", 1)[1] if " " in c else ""
                        keep = is_valid or self._is_compact_reference_token(suffix)
                    else:
                        keep = is_valid

                    if keep and (not cid or cl != cid):
                        confirmed.append(c)
            chunk['references'] = sorted(list(set(confirmed)))
            chunk['external_references'] = self._extract_external_standard_refs(text)
        return chunks

    def reconstruct_split_headers(self, lines):
        if not lines: return []
        merged = []
        i = 0
        while i < len(lines):
            curr = lines[i]
            txt = curr['text'].strip()

            # Check for short numeric or roman text
            is_num = len(txt) < 10 and re.match(r'^[\(\d][\d\.\)\w]+$', txt) and not re.search(r'[a-zA-Z]{3,}', txt)
            is_rom = re.match(r'^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|[A-Z])\.$', txt)

            if (is_num or is_rom) and (i + 1 < len(lines)):
                nxt = lines[i+1]
                nxt_txt = nxt['text'].strip()

                # Check if next line looks like a continuation
                is_list_item = re.match(r'^\s*\(?[\da-zA-Z]+\)', nxt_txt)

                if nxt_txt and (nxt_txt[0].isupper() or nxt_txt[0] in '"“\'' or nxt_txt[0].isdigit()) and not is_list_item:
                    # MERGE LOGIC UPDATED to include rich_spans
                    merged.append({
                        "text": f"{txt} {nxt_txt}",
                        "words": curr['words'] + nxt['words'],
                        "page": curr['page'],
                        "rich_spans": curr.get('rich_spans', []) + nxt.get('rich_spans', []) # <--- ADDED
                    })
                    i += 2
                    continue

            merged.append(curr)
            i += 1
        return merged

    def parse_structure(self, lines, patterns):
        chunks = []
        current = None
        seen = set()
        mode = 'default'
        toc_titles = set()
        toc_ids = set()
        parent_stack = [(-2, "ROOT")]

        # --- NEW FLAG: Track if we have entered the main document structure ---
        # This prevents Bold text on the cover page (PRELUDE) from splitting chunks.
        structure_started = False

        ranks = {
            'cfr_part_heading': -1, 'section_heading': -1, 'appendix_heading': -1, 'article_heading': -1,
            'preamble_heading': 0, 'cfr_section_heading': 0, 'supplementary_info': 0, 'roman_upper_heading': 0,
            'multi_level_heading': 0, 'multi_level_content_heading': 0,
            'fmvss_paragraph': 1, 'alpha_upper_heading': 1, 'definition_heading': 1,
            'implicit_heading': 2,
            'numeric_paren_heading': 2, 'alpha_paren_heading': 3, 'simple_list_item': 4,
            'roman_lower_paren_heading': 4, 'alpha_closing_paren_heading': 5,
        }

        toc_start = re.compile(patterns.get('toc_start_heading', r'^\s*CONTENTS\s*$'), re.IGNORECASE)

        patterns['multi_level_heading'] = r'^\s*(\d{1,3}(?:\.\d+)*)\.?\s+([A-Z][A-Z\s]{2,})$'
        patterns['multi_level_content_heading'] = r'^\s*(\d{1,3}(?:\.\d+)*\.)\s+(.+)$'

        heading_order = [
            'preamble_heading', 'cfr_part_heading', 'cfr_section_heading', 'fmvss_paragraph',
            'implicit_heading', 'section_heading', 'multi_level_heading', 'multi_level_content_heading',
            'appendix_heading', 'article_heading', 'supplementary_info',
            'roman_upper_heading', 'alpha_upper_heading', 'definition_heading',
            'numeric_paren_heading', 'alpha_paren_heading', 'roman_lower_paren_heading'
        ]
        heading_res = {k: re.compile(patterns[k]) for k in heading_order if k in patterns}

        def finalize(c):
            if c and c.get('content_verbatim', '').strip(): chunks.append(c)

        def norm_title(t):
            t = re.split(r'\s*\.{3,}', t)[0].strip()
            t = re.sub(r'[\s\d\(\)¹²³⁴⁵⁶⁷⁸⁹]+$', '', t).strip()
            m = re.search(r'[a-zA-Z]', t)
            return t[m.start():].lower().strip() if m else t.lower().strip()

        def is_visual_header(rich_spans):
            if not rich_spans: return False
            bold_count = sum(len(s['text']) for s in rich_spans if s['is_bold'])
            total_count = sum(len(s['text']) for s in rich_spans)
            if total_count == 0: return False
            return (bold_count / total_count) > 0.8

        i = 0
        while i < len(lines):
            line_obj = lines[i]
            text = line_obj['text']

            if mode == 'default':
                if toc_start.match(text):
                    structure_started = True # <--- TOC starts structure
                    finalize(current)
                    mode = 'in_toc'
                    toc_titles = set()
                    toc_ids = set()
                    current = {
                        "clause_id": "TOC", "title": "Table of Contents", "parent_id": "ROOT",
                        "content_verbatim": text+"\n", "content_words": list(line_obj['words'])
                    }

                else:
                    match, h_type = None, None
                    for k in heading_order:
                        if k not in heading_res: continue
                        m = heading_res[k].match(text)
                        if m:
                            match, h_type = m, k
                            break

                    if match:
                        structure_started = True # <--- Numbered Header starts structure
                        finalize(current)

                        raw_cid = match.group(1).strip().strip('.')
                        if h_type == 'implicit_heading':
                            slug = "_".join(raw_cid.split()[:4])
                            raw_cid = slug

                        rank = ranks.get(h_type, 99)
                        if h_type in ['multi_level_heading', 'multi_level_content_heading']:
                            rank = raw_cid.count('.')
                        elif h_type == 'fmvss_paragraph':
                            rank = raw_cid.count('.') + 1

                        while parent_stack and parent_stack[-1][0] >= rank: parent_stack.pop()

                        pid = parent_stack[-1][1] if parent_stack else "ROOT"
                        cid = raw_cid

                        parent_stack.append((rank, cid))
                        seen.add(cid)

                        detected_title = self._extract_title_from_line(line_obj.get('rich_spans', []), raw_cid)

                        current = {
                            "clause_id": cid,
                            "title": detected_title,
                            "parent_id": pid,
                            "level": len(parent_stack)-1,
                            "content_verbatim": text+"\n",
                            "content_words": list(line_obj['words'])
                        }

                    # Bold lines inside an already-structured clause are treated as
                    # body content, not chunk boundaries.  Creating a NIL chunk here
                    # severs the parent_id chain and orphans the text.
                    elif structure_started and is_visual_header(line_obj.get('rich_spans', [])):
                        if current is None:
                            current = {
                                "clause_id": "PRELUDE", "title": "NIL", "parent_id": "ROOT",
                                "level": 0, "content_verbatim": "", "content_words": []
                            }
                        current['content_verbatim'] += text + "\n"
                        current['content_words'].extend(line_obj['words'])

                    elif text.strip():
                        if current is None:
                            # This ensures the first chunk is named PRELUDE and collects all cover page text
                            current = {
                                "clause_id": "PRELUDE", "title": "NIL", "parent_id": "ROOT",
                                "level": 0, "content_verbatim": "", "content_words": []
                            }
                        current['content_verbatim'] += text + "\n"
                        current['content_words'].extend(line_obj['words'])

            elif mode == 'in_toc':
                current['content_verbatim'] += text + "\n"
                current['content_words'].extend(line_obj['words'])

                is_heading = False
                found_id = None
                for k in heading_order:
                    m = heading_res[k].match(text)
                    if m:
                        is_heading = True; found_id = m.group(1).strip().strip('.')
                        break
                if is_heading and found_id and found_id not in toc_ids:
                    toc_ids.add(found_id)
                elif is_heading and found_id in toc_ids:
                    finalize(current)
                    mode = 'default'
                    current = None
                    # Do NOT reset parent_stack here — pre-TOC heading context
                    # must be preserved so post-TOC headings get the right parent.
                    i -= 1

            i += 1
        finalize(current)
        return chunks

    def _merge_rects(self, rects, thresh=30):
        if not rects: return []
        rects.sort(key=lambda r: r.y0)
        merged = []
        while rects:
            curr = rects.pop(0)
            i = 0
            while i < len(rects):
                other = rects[i]
                expanded = fitz.Rect(curr.x0-thresh, curr.y0-thresh, curr.x1+thresh, curr.y1+thresh)
                if expanded.intersects(other):
                    curr |= other
                    rects.pop(i)
                else: i += 1
            merged.append(curr)
        return merged

    def _merge_image_bboxes(self, bboxes, gap=20):
        """Merge pixel-space bboxes [x1,y1,x2,y2] that overlap or are within `gap` pixels."""
        if not bboxes: return []
        boxes = [list(b) for b in bboxes]
        merged = True
        while merged:
            merged = False
            result, used = [], [False] * len(boxes)
            for i, a in enumerate(boxes):
                if used[i]: continue
                for j, b in enumerate(boxes):
                    if i == j or used[j]: continue
                    if a[0]-gap < b[2] and a[2]+gap > b[0] and a[1]-gap < b[3] and a[3]+gap > b[1]:
                        a = [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]
                        used[j] = True
                        merged = True
                result.append(a)
                used[i] = True
            boxes = result
        return boxes

    def extract_complex_elements(self, doc, chunks):
        page_map = defaultdict(list)
        for c in chunks:
            c['tables_html'] = []          # always reset tables (rebuilt fresh here)
            c['tables_bbox'] = []
            if 'images' not in c:          # IMPORTANT: do NOT reset images — they were
                c['images'] = []           # already attached upstream by hybrid_extract_and_structure
            if c.get('content_words'):
                page_map[c['content_words'][0]['page']-1].append({'chunk': c})

        for p_idx in range(len(doc)):
            chunk_data = page_map.get(p_idx, [])
            if not chunk_data: continue
            page = doc[p_idx]

            excl_rects = []
            try:
                for t in page.find_tables(strategy="lines"):
                    if t.row_count > 1 and t.col_count > 1:
                        page_chunks = [item['chunk'] for item in chunk_data]
                        closest = self._select_table_target_chunk(
                            page_chunks,
                            p_idx + 1,
                            table_bbox=t.bbox
                        )
                        if closest is None:
                            continue
                        df = t.to_pandas().map(lambda x: x.replace('\n', '<br>') if isinstance(x,str) else x)
                        prev_count = len(closest.get('tables_html', []))
                        closest['tables_html'] = self._merge_tables_html(
                            closest.get('tables_html', []),
                            [df.to_html(index=False, header=True, border=1, escape=False).replace('\n', '')],
                            prefer_incoming=True
                        )
                        if len(closest['tables_html']) > prev_count:
                            tbl_bbox = {str(p_idx + 1): [[round(t.bbox[0], 1), round(t.bbox[1], 1),
                                                           round(t.bbox[2], 1), round(t.bbox[3], 1)]]}
                            closest.setdefault('tables_bbox', []).append(tbl_bbox)
                        excl_rects.append(fitz.Rect(t.bbox))
            except: pass
            # Note: image extraction is now handled upstream in hybrid_extract_and_structure
            # via PaddleOCR bbox detection (merge + base64 crop), so the old
            # page.get_images() / page.get_drawings() sub-pass has been removed.
        return chunks

    def _try_convert_incorporating_to_html(self, content_verbatim):
        """
        Pure-code fallback: detect the interleaved 2-column amendment/date pattern
        that PyMuPDF produces when extracting a 2-column layout (e.g. the
        'Incorporating:' status page of ECE/UN regulations).

        Pattern: lines alternate between an amendment description and a date string
        (starting with 'Date of Entry into Force:' or 'Dated:').

        Returns an HTML table string if at least 3 pairs are found, else None.
        The chunk's content_verbatim is replaced with just the first (heading) line
        so the table carries the structured data.
        """
        date_pat = re.compile(
            r'^(Date\s+of\s+Entry\s+into\s+Force|Dated)\s*:', re.IGNORECASE
        )
        lines = [l.strip() for l in content_verbatim.splitlines() if l.strip()]
        if len(lines) < 4:
            return None

        pairs = []
        i = 0
        while i < len(lines):
            if date_pat.match(lines[i]):
                i += 1          # orphan date line — skip
                continue
            if i + 1 < len(lines) and date_pat.match(lines[i + 1]):
                pairs.append((lines[i], lines[i + 1]))
                i += 2
            else:
                i += 1          # amendment without a matching date — skip

        if len(pairs) < 3:
            return None

        rows_html = "".join(
            f"<tr><td>{amend}</td><td>{date}</td></tr>"
            for amend, date in pairs
        )
        return (
            '<table border="1" class="dataframe incorporating-dates">'
            '<thead><tr><th>Amendment</th><th>Date</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )

    # ── Artifact patterns: domain/URL/hex-hash strings that are never real headings ─
    _ARTIFACT_URL_RE = re.compile(
        r'(?i)(?:'
        r'\(?[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)+\.[a-z]{2,6}(?![a-z\-])\)?'  # (domain.tld) or domain.tld — TLD must not be mid-word (e.g. "TREATIES")
        r'|[a-f0-9]{8,}/[a-z0-9\-/]+'   # hex-hash/path  e.g. 62aaacee3a51/iso-20474-1-2017
        r'|https?://'                    # explicit URL
        r'|www\.[a-z]'                   # www.something
        r')'
    )

    # Distributor watermark phrases that appear verbatim across many pages.
    # These won't match _ARTIFACT_URL_RE (no domain/URL shape) so they get
    # their own pattern for line-level scrubbing.
    _ARTIFACT_PHRASE_RE = re.compile(
        r'(?i)(?:'
        r'iTeh\s+STANDARD\s+PREVIEW'     # iTeh preview banner
        r'|STANDARD\s+PREVIEW'           # without the brand
        r'|Licensed\s+to\s+\w'           # "Licensed to <name>" DRM stamps
        r')'
    )

    # Garbled circular-stamp noise: OCR reads rotating text as scattered single
    # chars separated by whitespace, producing lines like "( t", "d", "d it h i)".
    # Two heuristics cover both the very-short (1-2 token) and longer garbled cases.
    @staticmethod
    def _is_garbled_stamp_line(line: str) -> bool:
        tokens = line.split()
        if not tokens or len(line.strip()) > 30:
            return False
        alpha_lens = [len(re.sub(r'[^a-zA-Z0-9]', '', t)) for t in tokens]
        single_char_count = sum(1 for n in alpha_lens if n <= 1)
        # 1-2 token line where every token is a single char/symbol — pure noise
        if len(tokens) <= 2 and single_char_count == len(tokens):
            return True
        # 1-2 token line where no token has ≥ 3 alpha chars — garbled fragment
        # e.g. "Th C", "G lb" from OCR bbox clipping through a word boundary.
        if len(tokens) <= 2 and all(n <= 2 for n in alpha_lens):
            return True
        # Longer line where ≥ 60 % of tokens are single chars — garbled stamp
        return len(tokens) >= 3 and single_char_count / len(tokens) >= 0.6

    def _filter_pdf_artifacts(self, chunks):
        """
        Drop chunks that are PDF watermarks or distributor overlays (e.g. iTeh,
        standards.iteh.ai preview stamps).  Two independent signals — either
        alone is enough to discard a chunk:

          1. URL/domain pattern: clause_id or first content line looks like a
             website, URL fragment, or hex hash.  Real headings never look like
             that.

          2. Cross-page repetition: identical short text (≤ 12 words) appears
             on 4+ distinct pages, which is the hallmark of a repeating stamp.
             Legitimate section headings are unique across pages.

        After dropping artifact chunks:
          - Any surviving chunk whose parent_id pointed to a dropped chunk is
            re-parented to that dropped chunk's own parent (or ROOT).
          - Artifact lines embedded inside surviving chunks' content_verbatim
            are scrubbed line-by-line.
        """
        def _looks_like_structural_marker(text):
            """
            Preserve short structural identifiers that legitimately repeat across a
            document, especially TOC number boxes like "1." ... "11." that also
            reappear as real clause headings later in the PDF.
            """
            token = re.sub(r'\s+', ' ', str(text or '').strip())
            if not token:
                return False
            return bool(re.fullmatch(
                r'(?i)(?:'
                r'\d{1,3}(?:\.\d{1,3})*\.?'          # 1. / 3.2 / 10
                r'|[A-Z]\d+[A-Z]?\.?'                # A1 / B6a
                r'|\(?[ivxlcdm]{1,8}\)?\.?'          # i / iv / (i)
                r'|\(?[a-z]\)?\.?'                   # a / (a)
                r')',
                token,
            ))

        def _is_artifact_chunk(chunk, repeated_fps):
            cid = str(chunk.get('clause_id') or '')
            verbatim = (chunk.get('content_verbatim') or '').strip()
            first_line = verbatim.split('\n')[0].strip()
            if self._ARTIFACT_URL_RE.search(cid) or self._ARTIFACT_URL_RE.search(first_line):
                return True
            if _looks_like_structural_marker(cid) or _looks_like_structural_marker(first_line):
                return False
            if len(verbatim.split()) <= 12:
                fp = re.sub(r'\s+', ' ', verbatim.lower().strip())
                if fp in repeated_fps:
                    return True
            return False

        def _scrub_artifact_lines(text):
            """Remove individual lines that look like artifact/watermark text."""
            clean = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    clean.append(line)
                    continue
                if _looks_like_structural_marker(stripped):
                    clean.append(line)
                    continue
                if (self._ARTIFACT_URL_RE.search(stripped)
                        or self._ARTIFACT_PHRASE_RE.search(stripped)
                        or self._is_garbled_stamp_line(stripped)):
                    continue
                clean.append(line)
            return '\n'.join(clean).strip()

        # --- Pass 1: build repetition index (text fingerprint → page set) ---
        text_pages: dict = defaultdict(set)
        for chunk in chunks:
            verbatim = (chunk.get('content_verbatim') or '').strip()
            if not verbatim or len(verbatim.split()) > 12:
                continue
            fp = re.sub(r'\s+', ' ', verbatim.lower().strip())
            for pg_key in chunk.get('block_bboxes', {}).keys():
                text_pages[fp].add(str(pg_key))

        repeated_fps = {fp for fp, pages in text_pages.items() if len(pages) >= 3}

        # --- Pass 2: identify dropped chunks and record their clause_id → parent mapping ---
        # This lets us re-parent children of dropped chunks in one step.
        dropped_cid_to_parent: dict = {}
        for chunk in chunks:
            if _is_artifact_chunk(chunk, repeated_fps):
                cid = str(chunk.get('clause_id') or '')
                parent = str(chunk.get('parent_id') or 'ROOT')
                if cid and cid not in ('NIL', 'ROOT'):
                    dropped_cid_to_parent[cid] = parent
                if self.session_logger:
                    verbatim = (chunk.get('content_verbatim') or '').strip()
                    first_line = verbatim.split('\n')[0].strip()
                    self.session_logger.info(
                        f"[artifact-filter] dropped clause_id={cid!r} | {first_line[:80]!r}"
                    )

        # --- Pass 3: keep surviving chunks, fix parent_ids, scrub content lines ---
        kept = []
        for chunk in chunks:
            if _is_artifact_chunk(chunk, repeated_fps):
                continue

            # Re-parent if this chunk's parent was dropped
            pid = str(chunk.get('parent_id') or 'ROOT')
            if pid in dropped_cid_to_parent:
                new_parent = dropped_cid_to_parent[pid]
                if self.session_logger:
                    self.session_logger.info(
                        f"[artifact-filter] re-parented clause_id={chunk.get('clause_id')!r} "
                        f"from dropped {pid!r} → {new_parent!r}"
                    )
                chunk = dict(chunk)
                chunk['parent_id'] = new_parent

            # Scrub artifact lines embedded inside legitimate content
            verbatim = chunk.get('content_verbatim') or ''
            scrubbed = _scrub_artifact_lines(verbatim)
            if scrubbed != verbatim:
                chunk = dict(chunk) if not isinstance(chunk, dict) else chunk
                chunk['content_verbatim'] = scrubbed
                if self.session_logger:
                    self.session_logger.info(
                        f"[artifact-filter] scrubbed artifact lines from chunk clause_id={chunk.get('clause_id')!r}"
                    )

            kept.append(chunk)

        return kept

    def enrich_and_finalize(self, chunks, ext_table_pages=None, progress_bar=None):

        def _build_tables_field(chunk):
            html_list  = chunk.get('tables_html', [])
            bbox_list  = chunk.get('tables_bbox', [])
            merged_bbox = {}
            for entry in bbox_list:
                if entry:
                    for page, rects in entry.items():
                        merged_bbox.setdefault(page, []).extend(rects)
            return {"table_data": html_list, "bbox": merged_bbox}

        final = []
        curr_cont = "NIL"
        total = len(chunks) or 1
        chunk_id_counter = 1

        for i, c in enumerate(chunks):
            if progress_bar is not None:
                progress_bar.progress((i + 1) / total, text=f"Finalizing chunk {i + 1}/{total}")
            full = self._sanitize_chunk_content_verbatim(c.get('content_verbatim', '').strip())
            cid = c.get('clause_id', 'NIL')
            upper_cid = cid.upper()

            if not full and not c.get('tables_html') and not c.get('images'):
                continue

            # --- Issue 5 fix: detect interleaved 2-column amendment/date layout
            # and convert to an HTML table.  Only runs when AI extraction has not
            # already populated tables_html (AI result takes priority). ---
            if not c.get('tables_html'):
                inc_html = self._try_convert_incorporating_to_html(full)
                if inc_html:
                    c['tables_html'] = [inc_html]
                    # Keep only the heading line (e.g. "Incorporating:") as verbatim text
                    full = full.split('\n', 1)[0].strip() or full
            # ----------------------------------------------------------------

            if upper_cid.startswith(("ANNEX ", "APPENDIX ")) or upper_cid in ("ANNEX", "APPENDIX"):
                curr_cont = cid

            # Preserve the level set during extraction. Only fall back to
            # dot-count for chunks that never had a level assigned (e.g. legacy data).
            lvl = c.get('level')
            if lvl is None:
                lvl = 0 if cid in ['TOC', 'PRELUDE', 'NIL'] else cid.count('.') + 1

            raw_pages = [w['page'] for w in c.get('content_words', [])]
            if raw_pages:
                unique_pages = sorted(list(set(raw_pages)))
                page_str = ", ".join(map(str, unique_pages))
            else:
                page_str = ""

            raw_tt = c.get('text_type', 'text')
            refined_tt = self._refine_text_type(raw_tt, full, level=lvl)
            raw_title = c.get('title', 'NIL')
            if raw_title in ('NIL', '', None) and refined_tt == 'definition':
                # Quoted term followed immediately by “means” (e.g. “Air bag” means ...)
                _def_m = re.search(r'[“””]([^”””]{1,120})[“””]\s+means\b', full)
                if _def_m:
                    raw_title = _def_m.group(1).strip()
                else:
                    # Unquoted bold/italic term at start (e.g. Administrator means ...)
                    _unquoted_m = re.match(r'^\s*([A-Z][A-Za-z\s\(\)\-]{1,80})\s+means\b', full)
                    if _unquoted_m:
                        raw_title = _unquoted_m.group(1).strip()
            final.append({
                "chunk_id": chunk_id_counter,
                "clause_id": cid,
                "title": raw_title,
                "parent_id": c.get('parent_id', 'ROOT'),
                "annex_appendix": curr_cont,
                "level": lvl,
                "text_type": refined_tt,
                "bbox": c.get('block_bboxes', {}),
                "references": c.get('references', []),
                "external_references": c.get('external_references', []),
                "source_page": page_str,
                "content_verbatim": full,
                "tables_html": _build_tables_field(c),
                "images": c.get('images', []),
                "topic_label": c.get('topic_label', ''),
                "secondary_topic_label": c.get('secondary_topic_label', ''),
                "topic_confidence": c.get('topic_confidence', 0.0),
                "topic_reason": c.get('topic_reason', ''),
                "review_required": c.get('review_required', 'Yes'),
            })
            chunk_id_counter += 1

        return final

    def _resolve_parent_ids(self, chunks):
        """
        Post-finalize pass: converts parent_id from clause_id strings to the
        final integer chunk_ids so that parent_id is a true foreign key.

        Must run after enrich_and_finalize so chunk_ids are stable and any
        empty chunks that were dropped are no longer in the list.

        ROOT stays as the sentinel string 'ROOT' (no chunk represents it).
        Clause_ids that can't be matched (e.g. a dropped heading) fall back
        to 'ROOT' so children are never left with a dangling reference.

        Resolution is proximity-based: each child resolves to the nearest
        *preceding* chunk with the matching clause_id.  This prevents short
        clause_ids like "1" or "a" that repeat across articles from being
        resolved to their first-ever occurrence rather than the local one.
        """
        # Build ordered list of (list_index, chunk_id) per clause_id.
        cid_positions: dict = {}
        for i, chunk in enumerate(chunks):
            cid = chunk.get('clause_id', '')
            if cid and cid not in ('NIL', 'TOC', 'PRELUDE'):
                cid_positions.setdefault(cid, []).append((i, chunk['chunk_id']))

        for i, chunk in enumerate(chunks):
            pid = chunk.get('parent_id')
            if not pid or pid == 'ROOT':
                continue
            candidates = cid_positions.get(str(pid))
            if not candidates:
                chunk['parent_id'] = 'ROOT'
                continue
            # Pick the nearest candidate that appears before this chunk.
            resolved = 'ROOT'
            for idx, cid_chunk_id in candidates:
                if idx < i:
                    resolved = cid_chunk_id  # keep updating — last one before i wins
                else:
                    break
            # If no preceding candidate exists, fall back to the first one.
            if resolved == 'ROOT':
                resolved = candidates[0][1]
            chunk['parent_id'] = str(resolved)
            # ── Resolve debug: log every resolution where child chunk_id > 110 ─
            if self.session_logger and chunk.get('chunk_id', 0) > 110:
                self.session_logger.info(
                    f"[RESOLVE-DBG] chunk_id={chunk.get('chunk_id')} "
                    f"clause_id={chunk.get('clause_id')!r} "
                    f"pid_clause={pid!r} → resolved_chunk={resolved} "
                    f"candidates={candidates}"
                )
            # ─────────────────────────────────────────────────────────────────

        return chunks

    def _chain_prelude_parents(self, chunks):
        """
        Post-resolve pass: chain prelude chunks and nest top-level sections
        under the last prelude.

        After _resolve_parent_ids all prelude chunks have parent_id='ROOT' and
        main-body level-1 headings also sit at ROOT.  This pass:
          1. Chains consecutive prelude chunks so each becomes a child of the
             immediately preceding prelude (forming a title spine, e.g.
             "SAFETY GLASS — SPECIFICATION" → "( First Revision )").
          2. Re-parents any level-1 heading/subheading chunks after the last
             prelude that still point to ROOT so they nest under the last
             prelude chunk (the document title block becomes their logical root).

        Special case: consecutive figure/table label chunks (e.g. "Table I",
        "Table II") that appear in the same clause group are siblings, not
        parent-child.  When a label chunk is encountered and a previous label
        chunk exists in the current group, we reuse the previous label's
        parent instead of chaining from the intermediate table-content prelude.
        """
        def _get_tables_data(tbl):
            if isinstance(tbl, dict):
                return tbl.get('table_data', [])
            if isinstance(tbl, list):
                return tbl
            return []
        _fig_label_re = re.compile(
            r'(?i)^(figure|table|fig\.?)\s+[A-Z0-9][A-Z0-9\-–—\.]*\s*$'
        )
        last_prelude_id = None
        last_prelude_idx = -1
        # parent_id shared by all figure/table labels in the current group;
        # reset when a non-prelude structural chunk signals a new clause context.
        last_fig_label_parent_id = None

        for i, chunk in enumerate(chunks):
            if chunk.get('text_type') != 'prelude':
                # A real clause/heading chunk starts a new context — clear the
                # shared figure-label parent so the next label group is fresh.
                if chunk.get('text_type') not in ('header', 'footer'):
                    last_fig_label_parent_id = None
                continue

            _cv = (chunk.get('content_verbatim') or '').strip()
            _is_fig_label = bool(_fig_label_re.match(_cv))

            if last_prelude_id is not None:
                if _is_fig_label and last_fig_label_parent_id is not None:
                    # Sibling label: attach to the same parent as the previous label.
                    chunk['parent_id'] = str(last_fig_label_parent_id)
                else:
                    chunk['parent_id'] = str(last_prelude_id)

            if _is_fig_label:
                # Record this label's parent so the next label becomes its sibling.
                last_fig_label_parent_id = chunk.get('parent_id')
            elif not _get_tables_data(chunk.get('tables_html')):
                # Plain text prelude (not a label, not table content) — reset the
                # group so subsequent labels don't share a stale parent.
                last_fig_label_parent_id = None

            last_prelude_id = chunk['chunk_id']
            last_prelude_idx = i

        if last_prelude_id is None:
            return chunks

        for chunk in chunks[last_prelude_idx + 1:]:
            if (chunk.get('parent_id') == 'ROOT'
                    and chunk.get('text_type') in ('heading', 'subheading')
                    and chunk.get('level', 0) == 1):
                chunk['parent_id'] = str(last_prelude_id)

        return chunks

    def _resolve_reference_chunk_ids(self, chunks):
        """
        Converts references from clause_id strings to chunk_id integers.
        Must run after enrich_and_finalize and _resolve_parent_ids so chunk_ids are stable.
        A clause can span multiple chunks (e.g. across pages), so all matching chunk_ids are included.

        When multiple chunks share the same clause_id (e.g. "6" exists in both the main
        body and inside Annex 1), we prefer chunks whose Annex/Appendix context matches
        the source chunk. TOC-like entries (content with dot leaders "......") are also
        excluded as reference targets.
        """
        # Map: clause_id (lower) -> list of chunk dicts (not just IDs, so we can filter)
        cid_to_chunks = {}
        for chunk in chunks:
            cid = chunk.get('clause_id', '')
            if cid and cid not in ('NIL', 'TOC', 'PRELUDE'):
                cid_to_chunks.setdefault(cid.lower(), []).append(chunk)

        def _is_toc_like(chunk):
            return '....' in chunk.get('content_verbatim', '')

        def _pick_chunks(ref_cid, source_annex):
            candidates = cid_to_chunks.get(ref_cid, [])
            # Strip out TOC-style entries (dot leaders in content)
            candidates = [c for c in candidates if not _is_toc_like(c)]
            if not candidates:
                return []
            # Prefer candidates in the same annex context as the source chunk
            same_annex = [c for c in candidates if c.get('annex_appendix', 'NIL') == source_annex]
            return same_annex if same_annex else candidates

        for chunk in chunks:
            source_annex = chunk.get('annex_appendix', 'NIL')
            resolved = []
            for ref in chunk.get('references', []):
                picked = _pick_chunks(str(ref).lower(), source_annex)
                resolved.extend(c['chunk_id'] for c in picked)
            chunk['references'] = sorted(set(resolved))

        return chunks

    def _resolve_contextual_references(self, chunks):
        """
        Fix over-broad references produced by _resolve_reference_chunk_ids.

        The flat resolver maps a bare clause_id like "3" to every paragraph-3 in
        the entire document.  This pass re-reads each chunk's verbatim text and
        handles three qualified patterns:

          A) "Paragraphs 3 and 4 of Article 1"   — para-list THEN article
          B) "Article 1, Paragraphs 3 and 4"     — article THEN para-list
          C) "Paragraph 4 of this Article"        — the article that contains this chunk
          D) "Paragraph 4 of the said Article"   — back-reference to the most
                                                    recently named article in the
                                                    same chunk text
        """
        cid_upper = {}
        for chunk in chunks:
            cid = chunk.get('clause_id', '')
            if cid and cid not in ('NIL', 'TOC', 'PRELUDE'):
                cid_upper.setdefault(cid.upper(), []).append(chunk)

        # Build positional article containment: scan chunks in order and track
        # the last "ARTICLE N" heading seen. Paragraph chunks often have
        # parent_id="ROOT" rather than pointing to their article heading, so we
        # need this positional map as a fallback.
        chunks_sorted = sorted(chunks, key=lambda c: c['chunk_id'])
        containing_article_num = {}   # chunk_id -> article_num str (or None)
        _cur_art = None
        for c in chunks_sorted:
            cid_up = (c.get('clause_id') or '').strip().upper()
            m = re.match(r'^ARTICLE\s+(\d+)$', cid_up)
            if m:
                _cur_art = m.group(1)
            containing_article_num[c['chunk_id']] = _cur_art

        # Set of chunk_ids whose clause_id is an article heading ("ARTICLE N")
        article_heading_chunk_ids = {
            c['chunk_id'] for c in chunks
            if re.match(r'^ARTICLE\s+\d+$', (c.get('clause_id') or '').upper())
        }

        # Matches: "1", "1 and 2", "2, 3 and 5", "1 to 3"
        NUM_BLOB = r'[\d]+(?:(?:\s+to\s+|[\s,]+(?:and\s+)?)[\d]+)*'

        def _expand_para_nums(blob):
            """Return individual paragraph number strings from a blob, expanding N-to-M ranges."""
            range_m = re.search(r'(\d+)\s+to\s+(\d+)', blob)
            if range_m:
                start, end = int(range_m.group(1)), int(range_m.group(2))
                if start < end <= start + 20:
                    return [str(i) for i in range(start, end + 1)]
            return re.findall(r'\d+', blob)

        # Pattern A: "Paragraph(s) <nums> of [the previous version of] Article N"
        pat_a = re.compile(
            r'(?i)\bparagraphs?\s+(' + NUM_BLOB + r')'
            r'(?:\s+of\s+(?:the\s+)?(?:\w+\s+){0,4}?Article\s+(\d+))'
        )
        # Pattern B: "Article N, Paragraph(s) <nums>"
        pat_b = re.compile(
            r'(?i)\bArticle\s+(\d+)[,\s]+Paragraphs?\s+(' + NUM_BLOB + r')'
        )
        # Pattern C: "Paragraph(s) <nums> of this Article" — current chunk's article
        pat_c_this = re.compile(
            r'(?i)\bparagraphs?\s+(' + NUM_BLOB + r')'
            r'\s+of\s+this\s+Article\b'
        )
        # Pattern D: "Paragraph(s) <nums> of the said Article" — last named article
        pat_c_said = re.compile(
            r'(?i)\bparagraphs?\s+(' + NUM_BLOB + r')'
            r'\s+of\s+the\s+said\s+Article\b'
        )
        # All "Article N" mentions, for resolving back-references
        pat_art_num = re.compile(r'(?i)\bArticle\s+(\d+)\b')

        def _children_for(article_num, para_nums):
            """Return chunk_ids belonging to ARTICLE <article_num> with clause_id in para_nums."""
            article_key = f"ARTICLE {article_num}"
            art_chunks = cid_upper.get(article_key, [])
            if not art_chunks:
                return []
            # Try parent_id-based lookup first
            art_ids = {str(ac['chunk_id']) for ac in art_chunks}
            found = [
                c['chunk_id'] for c in chunks
                if str(c.get('parent_id')) in art_ids and c.get('clause_id') in para_nums
            ]
            if found:
                return found
            # Fallback: positional containment (parent_id may be ROOT for flat structures)
            return [
                c['chunk_id'] for c in chunks_sorted
                if containing_article_num.get(c['chunk_id']) == article_num
                and c.get('clause_id') in para_nums
            ]

        for chunk in chunks:
            text = chunk.get('content_verbatim', '')
            contextual_ids = []

            # Pattern A
            for m in pat_a.finditer(text):
                para_blob, article_num = m.group(1), m.group(2)
                if not article_num:
                    continue
                contextual_ids.extend(_children_for(article_num, _expand_para_nums(para_blob)))

            # Pattern B
            for m in pat_b.finditer(text):
                article_num, para_blob = m.group(1), m.group(2)
                contextual_ids.extend(_children_for(article_num, _expand_para_nums(para_blob)))

            # Pattern C — "this Article" means the article containing this chunk
            for m in pat_c_this.finditer(text):
                para_blob = m.group(1)
                article_num = containing_article_num.get(chunk['chunk_id'])
                if article_num:
                    contextual_ids.extend(_children_for(article_num, _expand_para_nums(para_blob)))

            # Pattern D — "the said Article" → last Article N mentioned before this phrase
            for m in pat_c_said.finditer(text):
                para_blob = m.group(1)
                article_num = None
                for art_m in pat_art_num.finditer(text):
                    if art_m.start() < m.start():
                        article_num = art_m.group(1)
                if article_num:
                    contextual_ids.extend(_children_for(article_num, _expand_para_nums(para_blob)))

            if contextual_ids:
                # Keep article-heading refs that were correctly resolved earlier;
                # replace only the over-broad bare-paragraph refs.
                preserved = article_heading_chunk_ids & set(chunk.get('references', []))
                chunk['references'] = sorted(set(contextual_ids) | preserved)

        return chunks

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def _save_checkpoint(self, path, p_idx, all_chunks, raw_chunks,
                         all_page_images, state, metrics):
        """
        Serialize the full mid-run state to a JSON file so processing can
        resume after a crash.  Called every 10 pages (aligned with GC cleanup).

        New pipeline state: raw_chunks (block-level, unannotated) replaces the
        old current_chunk accumulator.  parent_stack / annex_context are no
        longer needed mid-loop because those passes run post-loop.
        """
        def rect_to_list(r):
            return [r.x0, r.y0, r.x1, r.y1]

        serializable_metrics = {
            "total_words":               metrics.get("total_words", 0),
            "kept_words":                metrics.get("kept_words", 0),
            "dropped_words":             metrics.get("dropped_words", 0),
            "removed_header_lines":      metrics.get("removed_header_lines", 0),
            "removed_footer_lines":      metrics.get("removed_footer_lines", 0),
            "removed_edge_line_samples": metrics.get("removed_edge_line_samples", []),
            "detected_tables": {
                str(pg): [rect_to_list(r) for r in rects]
                for pg, rects in metrics.get("detected_tables", {}).items()
            },
            "detected_columnar_content": {
                str(pg): [rect_to_list(r) for r in rects]
                for pg, rects in metrics.get("detected_columnar_content", {}).items()
            },
        }
        payload = {
            "last_completed_page": p_idx,
            "all_chunks":          all_chunks,
            "raw_chunks":          raw_chunks,       # NEW: replaces current_chunk
            "all_page_images":     all_page_images,
            "state": {
                "chunk_counter": state["chunk_counter"],
                "image_counter": state["image_counter"],
                # parent_stack / annex_context / toc state removed —
                # these are recomputed by post-loop passes on full raw_chunks
            },
            "metrics": serializable_metrics,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            if self.session_logger:
                self.session_logger.warning(f"[CHECKPOINT SAVE FAILED] {e}")

    def _load_checkpoint(self, path):
        """
        Load a previously saved checkpoint.
        Converts rect lists back to fitz.Rect objects.
        Returns the payload dict, or None if missing / corrupt.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            def lists_to_rects(d):
                return {
                    int(pg): [fitz.Rect(*coords) for coords in rects]
                    for pg, rects in d.items()
                }

            data["metrics"]["detected_tables"] = lists_to_rects(
                data["metrics"].get("detected_tables", {}))
            data["metrics"]["detected_columnar_content"] = lists_to_rects(
                data["metrics"].get("detected_columnar_content", {}))
            # Backward-compat: old checkpoints have current_chunk instead of raw_chunks
            if "raw_chunks" not in data:
                data["raw_chunks"] = []
            return data
        except Exception as e:
            if self.session_logger:
                self.session_logger.warning(f"[CHECKPOINT LOAD FAILED] {e}")
            return None
    # ─────────────────────────────────────────────────────────────────────────

    def generate_ai_summary(self, chunks, pdf_bytes=None, model="gpt-4o"):
        """
        Generate a structured 5-section regulatory briefing using RAG + GPT-4o.
        Uses raw PDF text extracted via fitz, chunked and embedded with sentence-transformers,
        then retrieved per-section and synthesised by the LLM.
        Returns a markdown string, or "" on failure.
        """
        if not self._has_ai() or not pdf_bytes:
            return ""

        # ── Section retrieval queries (one set per customer output section) ──
        _SECTION_QUERIES = {
            "regulatory_overview": [
                "title scope purpose document type regulation supplement corrigendum amendment revision replacement consolidated version",
                "parent regulation modifies replaces supplement series revision number implementing act delegated act repeal",
                "objective overview this regulation applies to vehicles requirements approval framework automotive",
            ],
            "purpose_and_context": [
                "purpose objective rationale intended to ensure safety environmental operational concerns risk problem addresses",
                "exists to ensure prevent reduce improve protect manage failures hazards driver system environment",
            ],
            "scope_and_applicability": [
                "scope application applies to vehicle categories vehicles manufacturers stakeholders systems components operational conditions",
                "contracting parties geographic applicability vehicle types affected systems components driver software sensor braking steering data",
            ],
            "key_themes_and_obligations": [
                "manufacturers shall requirements obligations approval certification conformity production evidence testing governance documentation",
                "software updates cybersecurity data recording monitoring transition demand markings labelling market surveillance fault handling",
            ],
            "compliance_timeline": [
                "entry into force date dates transitional provisions mandatory from until application deadlines approvals granted before after",
                "new vehicle types existing vehicle types compliance date implementation period transitional arrangements expiry sunset",
            ],
        }

        _BRIEFING_PROMPT = """You are drafting the opening section of a professional automotive regulatory briefing for business and engineering readers.

Your task is to produce an executive-facing preface summary that helps a reader quickly understand:
- what the regulation is,
- why it exists,
- what it impacts,
- and why it matters.

STRICT RULES:
- Use ONLY information explicitly supported by the provided context.
- Do NOT introduce external knowledge.
- Do NOT hallucinate, speculate, or assume hidden intent.
- Do NOT quote large sections verbatim.
- Do NOT produce a clause-by-clause breakdown.
- Summarize in clear, concise, professional language.
- Focus on business and engineering understanding rather than legal drafting style.
- Maintain a neutral, executive-ready tone.
- When a detail is not explicitly available, write exactly: "Not explicitly identified in the provided text."
- For likely impacted vehicle systems, mention them only where the text directly supports that linkage or where the system is clearly and explicitly referenced in the regulation context.
- Prefer structured paragraphs. Use bullets only sparingly where they improve readability.
- Target approximately 600-1200 words.

STYLE TARGET:
- The output should read like the opening section of a professional regulatory briefing document.
- It should feel information-rich, disciplined, and commercially useful.
- It should explain significance and enterprise implications without sounding promotional.

Return the output in exactly this order and with exactly these section headings:

1. Regulatory Overview and Context
2. Purpose and Context
3. Scope and Applicability
4. Key Themes and Obligations
5. Compliance Timeline

SECTION-SPECIFIC INSTRUCTIONS:

1. Regulatory Overview and Context
- Identify whether the document appears to be a new regulation, amendment, supplement, corrigendum, revision, replacement, consolidated version, delegated / implementing act, or repeal, but only if explicitly supported.
- If applicable, identify the parent regulation, amendment or revision number, what regulation or version it modifies or replaces, and whether the changes appear technical, administrative, editorial, or procedural, but only if explicitly supported.
- Then explain at a high level what the regulation covers, its overall purpose, why it is significant to the automotive industry, and the likely enterprise-level impact.
- Where any of these are not explicitly available, state: "Not explicitly identified in the provided text."

2. Purpose and Context
- Explain why the regulation exists based only on the text.
- Summarize the risks, safety concerns, environmental concerns, or operational issues it addresses.
- Explain what regulatory or industry problem it appears intended to solve.
- Do not invent rationale beyond what is directly supported.

3. Scope and Applicability
- Describe vehicle types affected, systems or components affected, stakeholders impacted, operational scope, and geographic applicability if identifiable.
- Mention likely impacted vehicle systems only where the text clearly supports them.
- If any of these are not explicit, state: "Not explicitly identified in the provided text."

4. Key Themes and Obligations
- Summarize the major regulatory themes and obligations without listing every clause.
- Focus on what manufacturers are required to do, what evidence or governance may be required, and what operational or engineering capabilities are implied.
- Keep this section thematic and readable.

5. Compliance Timeline
- Identify and summarize all explicit regulatory dates and timing provisions found in the context.
- Include entry into force dates, mandatory compliance dates, transitional provisions, applicability dates for new or existing vehicle types, implementation deadlines, approval timing provisions, and expiry or sunset provisions where explicitly present.
- For each identified date or timing provision, explain what it represents, who or what it applies to, and why it matters from a business standpoint.
- If none are present, write exactly: "No explicit compliance or applicability dates were identified in the provided text."

CONTEXT:
{context}""".strip()

        try:
            # ── Step 1: Extract raw text page-by-page via fitz ───────────────
            _doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for _pnum, _page in enumerate(_doc, start=1):
                _text = _page.get_text("text") or ""
                _text = re.sub(r"[ \t]+", " ", _text)
                _text = re.sub(r"\n{3,}", "\n\n", _text).strip()
                if _text:
                    pages.append({"page_num": _pnum, "text": _text})
            _doc.close()

            if not pages:
                return ""

            total_pages = len(pages)
            total_chars = sum(len(p["text"]) for p in pages)

            # ── Step 2: Adaptive retrieval plan based on document size ────────
            if total_pages <= 10 or total_chars <= 30_000:
                top_k, window, budget, chunk_size, overlap = 4, 1, 12, 1200, 250
            elif total_pages <= 30 or total_chars <= 120_000:
                top_k, window, budget, chunk_size, overlap = 6, 2, 20, 1400, 300
            elif total_pages <= 75 or total_chars <= 300_000:
                top_k, window, budget, chunk_size, overlap = 8, 2, 28, 1600, 350
            else:
                top_k, window, budget, chunk_size, overlap = 10, 3, 36, 1800, 400

            # ── Step 3: Chunk raw text (paragraph-aware with character limit) ─
            def _chunk_text(text, size, ovlp):
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                result_chunks, current = [], ""
                for para in paragraphs:
                    candidate = f"{current}\n\n{para}".strip() if current else para
                    if len(candidate) <= size:
                        current = candidate
                    else:
                        if current:
                            result_chunks.append(current)
                        if len(para) <= size:
                            current = para
                        else:
                            step = max(size - ovlp, 1)
                            for start in range(0, len(para), step):
                                piece = para[start:start + size].strip()
                                if piece:
                                    result_chunks.append(piece)
                            current = ""
                if current:
                    result_chunks.append(current)
                return result_chunks

            rows = []
            seq = 0
            for page in pages:
                for chunk_text in _chunk_text(page["text"], chunk_size, overlap):
                    seq += 1
                    rows.append({
                        "seq":          seq,
                        "page_num":     page["page_num"],
                        "chunk_id":     f"p{page['page_num']}::c{seq}",
                        "source_label": f"p.{page['page_num']}",
                        "text":         chunk_text,
                    })

            if not rows:
                return ""

            # ── Step 4: Embed all chunks with the cached sentence transformer ─
            _model = load_summary_model()
            if _model is None:
                return ""

            _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
            embeddings = np.array(
                _model.encode(
                    [r["text"] for r in rows],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                ),
                dtype=np.float32,
            )

            seq_to_idx = {r["seq"]: i for i, r in enumerate(rows)}

            def _embed_query(q):
                emb = _model.encode(
                    [f"{_QUERY_PREFIX}{q}"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return np.array(emb[0], dtype=np.float32)

            def _search(query, k):
                q_emb = _embed_query(query)
                scores = embeddings @ q_emb
                top_idx = np.argsort(scores)[::-1][:k]
                hits = []
                for idx in top_idx:
                    item = dict(rows[idx])
                    item["score"] = float(scores[idx])
                    hits.append(item)
                return hits

            def _expand(hits):
                by_seq = {r["seq"]: r for r in rows}
                expanded = {}
                for hit in hits:
                    center = hit["seq"]
                    for s in range(max(1, center - window), center + window + 1):
                        if s in by_seq:
                            row = dict(by_seq[s])
                            row["seed_score"] = hit.get("score", 0.0)
                            expanded[row["chunk_id"]] = row
                return list(expanded.values())

            def _rerank(candidates, query, k):
                q_emb = _embed_query(query)
                scored = []
                for cand in candidates:
                    idx = seq_to_idx[cand["seq"]]
                    item = dict(cand)
                    item["final_score"] = float(embeddings[idx] @ q_emb)
                    scored.append(item)
                scored.sort(key=lambda x: x["final_score"], reverse=True)
                return scored[:k]

            # ── Step 5: Per-section retrieve → expand → rerank ───────────────
            all_queries = [
                (section, q)
                for section, qs in _SECTION_QUERIES.items()
                for q in qs
            ]
            per_query_quota = max(1, budget // len(all_queries))

            query_groups = []
            for section, query in all_queries:
                hits     = _search(query, top_k)
                expanded = _expand(hits)
                ranked   = _rerank(expanded, query, per_query_quota)
                query_groups.append((section, query, ranked))

            # ── Step 6: Round-robin merge (equal section representation) ──────
            chosen, seen_ids = [], set()
            round_idx = 0
            while True:
                added_any = False
                for _, _, ranked_chunks in query_groups:
                    if round_idx < len(ranked_chunks):
                        item = ranked_chunks[round_idx]
                        if item["chunk_id"] not in seen_ids:
                            chosen.append(item)
                            seen_ids.add(item["chunk_id"])
                            added_any = True
                if not added_any:
                    break
                round_idx += 1

            # Backfill if deduplication left us short of budget
            if len(chosen) < budget:
                for _, query in all_queries:
                    for item in _rerank(_expand(_search(query, top_k + 2)), query, budget):
                        if item["chunk_id"] not in seen_ids:
                            chosen.append(item)
                            seen_ids.add(item["chunk_id"])
                            if len(chosen) >= budget:
                                break
                    if len(chosen) >= budget:
                        break

            chosen = chosen[:budget]

            # ── Step 7: Build context and call LLM ───────────────────────────
            context = "\n\n".join(
                f"[Source {i}] {item['source_label']}\n{item['text']}"
                for i, item in enumerate(chosen, start=1)
            )

            briefing = self._call_llm(
                model=model,
                system_prompt="You are a precise regulatory briefing writer. Use only the provided context. Do not hallucinate.",
                user_prompt=_BRIEFING_PROMPT.format(context=context),
                temperature=0.1,
            )

            # ── Step 8: Extract metadata fields using existing embeddings ─────
            # regulation_number — regex over first 5 pages
            _head = " ".join(p["text"] for p in pages[:5])
            _reg_pats = [
                re.compile(
                    r'\b(?:ISO|IEC|EN|BS|ASTM|ANSI|DIN|JIS|NFPA)'
                    r'(?:\s*/\s*(?:TS|TR|PAS|IEC|DIS|FDIS))?'
                    r'\s*[\d]{3,6}(?:[\-:]\d{1,4}(?:[\-:]\d{1,4})?)?',
                    re.IGNORECASE,
                ),
                re.compile(
                    r'\b(?:Regulation|Directive|Decision|Ordinance|Act|Order|Rule)'
                    r'(?:\s+\((?:EU|EC|EEC|UK|US)\))?'
                    r'(?:\s+No\.?)?\s*[\d]{4}/[\d]{1,4}(?:/(?:EU|EC|EEC|UK))?',
                    re.IGNORECASE,
                ),
                re.compile(r'\b\d{1,3}\s+(?:CFR|U\.S\.C\.)\s+(?:Part\s+)?\d+', re.IGNORECASE),
            ]
            regulation_number = None
            for pat in _reg_pats:
                m = pat.search(_head)
                if m:
                    regulation_number = m.group(0).strip()
                    break

            # geography — first sentence with an explicit geo mention
            _geo_pat = re.compile(
                r'\b(?:European Union|EU member states?|United Kingdom|United States|'
                r'China|Japan|India|Australia|Canada|Germany|France|'
                r'international|worldwide|global|all countries|member states?)\b',
                re.IGNORECASE,
            )
            _incomplete_end = re.compile(r'(?:e\.g\.|i\.e\.|etc\.|,|:)\s*$', re.IGNORECASE)
            geography = None
            for row in rows:
                for sent in re.split(r'(?<=[.!?])\s+', row["text"]):
                    sent = sent.strip()
                    if _geo_pat.search(sent) and not _incomplete_end.search(sent) and len(sent) > 30:
                        geography = sent
                        break
                if geography:
                    break
            if geography is None and regulation_number:
                if re.match(r'(?:ISO|IEC|EN)\b', regulation_number, re.IGNORECASE):
                    geography = "International (ISO/IEC standard — applies globally)"

            # scope — cosine sim with scope queries, pick best non-negative sentence
            _scope_queries = [
                "this document specifies requirements for",
                "scope this standard applies to covers machinery equipment",
                "this regulation establishes rules obligations for",
            ]
            _scope_embs = np.array(
                _model.encode(_scope_queries, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            _scope_sims = (_scope_embs @ embeddings.T).max(axis=0)
            _neg_pat = re.compile(r'\bnot\s+(?:apply|cover|provide|include)\b|\bexclud', re.IGNORECASE)
            scope = None
            for idx in _scope_sims.argsort()[::-1][:15]:
                for sent in re.split(r'(?<=[.!?])\s+', rows[idx]["text"]):
                    sent = sent.strip()
                    if not _neg_pat.search(sent) and not _incomplete_end.search(sent) and len(sent.split()) >= 8:
                        scope = sent
                        break
                if scope:
                    break
            if scope is None:
                scope = rows[int(_scope_sims.argmax())]["text"][:200]

            # key_points — sentences closest to document centroid
            _centroid = embeddings.mean(axis=0)
            _centroid_scores = embeddings @ _centroid
            _used = {scope or "", geography or ""}
            key_points = []
            for idx in _centroid_scores.argsort()[::-1]:
                for sent in re.split(r'(?<=[.!?])\s+', rows[idx]["text"]):
                    sent = sent.strip()
                    if sent in _used or len(sent.split()) < 8 or _incomplete_end.search(sent):
                        continue
                    key_points.append(sent)
                    _used.add(sent)
                    break
                if len(key_points) == 5:
                    break

            del embeddings
            gc.collect()

            return {
                "regulation_number": regulation_number,
                "geography":         geography,
                "scope":             scope,
                "key_points":        key_points,
                "briefing":          briefing or "",
            }

        except Exception as _e:
            if self.session_logger:
                self.session_logger.warning(f"[SUMMARY FAILED] {type(_e).__name__}: {_e}")
            return {}

    def run_pipeline(self, pdf_bytes, num_pages, enable_ai, ai_thresh, table_model, analysis_model, api_key, paddle_model, model_type, ai_provider="openai", checkpoint_path=None, resume_from_page=0, page_start=0, progress_state=None, enable_topic_ai=True, enable_ai_summary=True):
        self.openai_api_key = api_key
        self.ai_api_key = api_key
        self.ai_provider = ai_provider

        # ── Session logger setup ──────────────────────────────────────────────
        pdf_name = self.document_name
        self.session_logger = _create_session_logger(pdf_name)
        _session_t0 = time.time()
        self.session_logger.info(
            f"[SESSION START] file={pdf_name!r} pages={num_pages} model={model_type} "
            f"ai_tables={enable_ai} ai_topic={enable_topic_ai} provider={ai_provider} "
            f"resume_page={resume_from_page}"
        )
        # ─────────────────────────────────────────────────────────────────────

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            if self.session_logger:
                self.session_logger.error(f"[PDF OPEN ERROR] {type(e).__name__}: {e}")
            st.error(f"PDF Error: {e}")
            return [], None, None, None

        try:
         _pipeline_interrupted = False
         with st.status("🚀 Processing...", expanded=True) as status:

            # --- VISION PROGRESS BAR ---
            st.write("👁️ Running Vision Layout Analysis & Structure Logic...")
            vision_bar = st.progress(0, text="Initializing Vision Scan...")

            _effective_start = resume_from_page if resume_from_page > 0 else page_start
            _total_pages_label = num_pages  # end page (1-indexed)
            def update_vision_bar(progress):
                current_page = _effective_start + int(progress * (num_pages - _effective_start))
                vision_bar.progress(progress, text=f"Scanning Page {current_page}/{_total_pages_label}")
                if progress_state is not None:
                    progress_state["page"] = current_page

            chunks, metrics = self.hybrid_extract_and_structure(
                doc, num_pages, paddle_model, model_type,
                progress_callback=update_vision_bar,
                status_callback=lambda text: vision_bar.progress(0, text=text),
                checkpoint_path=checkpoint_path,
                resume_from_page=resume_from_page,
                page_start=page_start,
            )
            vision_bar.empty() # Remove bar when done
            # ---------------------------

            st.write("🔗 Resolving structure & cross-references...")
            ref_bar = st.progress(0, text="Extracting complex elements...")
            chunks = self.extract_complex_elements(doc, chunks)
            ref_bar.progress(0.4, text="Resolving references...")
            chunks = self._resolve_references(chunks)
            ref_bar.progress(1.0, text="Structure resolved.")
            ref_bar.empty()

            ext_pages = set()

            # 4. AI Table Extraction (Targeted via Paddle Detection)
            if enable_ai and api_key and metrics['detected_tables']:
                total_tables = sum(len(v) for v in metrics['detected_tables'].values())
                st.write(f"📊 AI Extracting {total_tables} detected tables...")

                # --- TABLE PROGRESS BAR ---
                table_bar = st.progress(0, text="Queueing tables...")
                # --------------------------

                te = TableExtractor(api_key, model=table_model, provider=ai_provider, session_logger=self.session_logger)
                tasks = []
                for p_num, rects in metrics['detected_tables'].items():
                    for r in rects:
                        tasks.append((doc[p_num-1], p_num, r))

                results = {}
                completed_tables = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_task = {executor.submit(te._extract_page_sync, t): t for t in tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        # --- UPDATE TABLE BAR ---
                        completed_tables += 1
                        table_bar.progress(completed_tables / total_tables, text=f"Extracting Table {completed_tables}/{total_tables}")
                        # ------------------------
                        try:
                            _task = future_to_task[future]
                            _rect = _task[2] if len(_task) >= 3 else None
                            p_num, tables = future.result()
                            if tables:
                                if p_num not in results: results[p_num] = []
                                results[p_num].extend(tables)
                                html_list = [t.get('html') for t in tables if t.get('html')]
                                self._attach_ai_tables_to_chunks_by_page(
                                    chunks, p_num, html_list, rect=_rect
                                )
                        except Exception: pass

                table_bar.empty() # Remove bar when done
                ext_pages = set(results.keys())
                st.session_state['ai_table_report_data'] = results

            # 5. Optional AI extraction for columnar CONTENT blocks (e.g. "Incorporating:"
            #    dates tables detected by PaddleOCR as CONTENT, not TABLE).
            #    Results are injected into chunk['tables_html'] BEFORE enrich_and_finalize
            #    so the pure-code fallback is skipped for those chunks (AI is more accurate).
            if enable_ai and api_key and metrics.get('detected_columnar_content'):
                col_tasks = []
                for p_num, rects in metrics['detected_columnar_content'].items():
                    for r in rects:
                        col_tasks.append((doc[p_num - 1], p_num, r))
                if col_tasks:
                    st.write(f"📋 AI Extracting {len(col_tasks)} columnar content block(s)...")
                    col_te = TableExtractor(api_key, model=table_model, provider=ai_provider, session_logger=self.session_logger)
                    col_bar = st.progress(0, text="Extracting columnar content...")
                    completed_col = 0
                    # Build a page→chunk lookup using source_page stored on each chunk
                    page_to_chunks = defaultdict(list)
                    for chunk in chunks:
                        for pg_str in chunk.get('source_page', '').split(','):
                            pg_str = pg_str.strip()
                            if pg_str.isdigit():
                                page_to_chunks[int(pg_str)].append(chunk)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        col_futures = {executor.submit(col_te._extract_page_sync, t): t[1] for t in col_tasks}
                        for future in concurrent.futures.as_completed(col_futures):
                            completed_col += 1
                            col_bar.progress(completed_col / len(col_tasks),
                                             text=f"Columnar block {completed_col}/{len(col_tasks)}")
                            try:
                                p_num, tables = future.result()
                                if tables:
                                    target_chunks = page_to_chunks.get(p_num, [])
                                    if target_chunks:
                                        target_chunk = target_chunks[0]
                                        incoming_html = [t.get('html') for t in tables if t.get('html')]
                                        target_chunk['tables_html'] = self._dedupe_tables_html(incoming_html)
                            except Exception:
                                pass
                    col_bar.progress(1.0, text=f"✅ {len(col_tasks)} columnar block(s) extracted.")

            chunks = self._filter_pdf_artifacts(chunks)
            self._log_toc_debug_snapshot("post-artifact-filter", chunks)
            st.write("📝 Finalizing & enriching chunks...")
            total_chunks = len(chunks)
            enrich_bar = st.progress(0, text="Finalizing chunks...")
            final = self.enrich_and_finalize(chunks, ext_pages, progress_bar=enrich_bar)
            enrich_bar.progress(1.0, text=f"✅ {len(final)} chunks finalized.")
            enrich_bar.empty()

            final = self._resolve_parent_ids(final)
            final = self._chain_prelude_parents(final)
            final = self._resolve_reference_chunk_ids(final)
            final = self._resolve_contextual_references(final)
            final = self._fix_cross_page_table_continuation(final)

            st.write("🏷️ Assigning topic labels...")
            topic_bar = st.progress(0, text="Labeling topics...")
            final, topic_stats = self.assign_topic_labels(final, model=analysis_model, use_llm=enable_topic_ai, progress_bar=topic_bar)
            topic_bar.progress(1.0, text=f"✅ Topics assigned.")
            topic_bar.empty()
            metrics["topic_labeling"] = topic_stats

            st.write("📌 Assigning requirement labels...")
            req_bar = st.progress(0, text="Labeling requirement types...")
            final, req_stats = self.assign_requirement_labels(final, model=analysis_model, use_llm=enable_topic_ai, progress_bar=req_bar)
            req_bar.progress(1.0, text="✅ Requirement labels assigned.")
            req_bar.empty()
            metrics["requirement_labeling"] = req_stats

            if enable_topic_ai and self._has_ai():
                st.write("🧠 Assigning text type labels...")
                text_type_bar = st.progress(0, text="Labeling text types...")
                final = self.assign_text_type_labels(final, model=analysis_model, progress_bar=text_type_bar)
                text_type_bar.progress(1.0, text="✅ Text type labels assigned.")
                text_type_bar.empty()

            for c in final:
                c.pop('requirement_label', None)
                if not c.get('requirement_label_1'):
                    c['requirement_label_1'] = "Unclassified"
                if 'requirement_label_2' not in c:
                    c['requirement_label_2'] = None
                if 'ai_label_text_type' not in c: c['ai_label_text_type'] = "Unclassified"

            status.update(label="✅ Complete!", state="complete", expanded=False)

        except Exception as _pipeline_err:
            _pipeline_interrupted = True
            if self.session_logger:
                self.session_logger.error(
                    f"[SESSION INTERRUPTED] pipeline-level error: "
                    f"{type(_pipeline_err).__name__}: {_pipeline_err}"
                )
            raise

        # Delete checkpoint on successful completion — it is no longer needed
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass

        # ── Session log — final summary line ─────────────────────────────────
        if self.session_logger:
            _elapsed = time.time() - _session_t0
            _n_tables = sum(len(v) for v in metrics.get("detected_tables", {}).values())
            _n_col    = sum(len(v) for v in metrics.get("detected_columnar_content", {}).values())
            self.session_logger.info(
                f"[SESSION END] total_time={_elapsed:.1f}s chunks={len(final)} "
                f"words_kept={metrics.get('kept_words', 0)} "
                f"words_total={metrics.get('total_words', 0)} "
                f"tables_detected={_n_tables} columnar_detected={_n_col} "
                f"topics_rule={metrics.get('topic_labeling', {}).get('rule_labeled', 0)} "
                f"topics_llm={metrics.get('topic_labeling', {}).get('llm_labeled', 0)} "
                f"topics_review={metrics.get('topic_labeling', {}).get('review_required', 0)} "
                f"headers_removed={metrics.get('removed_header_lines', 0)} "
                f"footers_removed={metrics.get('removed_footer_lines', 0)} "
                f"log={self.session_logger._log_path!r}"
            )
            for _h in self.session_logger.handlers:
                _h.flush()
                _h.close()
        # ─────────────────────────────────────────────────────────────────────

        ai_summary = {}
        if enable_ai_summary and self._has_ai():
            with st.spinner("Generating document summary..."):
                ai_summary = self.generate_ai_summary(final, pdf_bytes=pdf_bytes, model=analysis_model)

        return final, metrics, None, ai_summary
