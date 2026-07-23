"""Importable PaddleOCR subprocess entry point.

Streamlit executes application scripts as a dynamic ``__main__`` module. A
``multiprocessing`` process using the safe CUDA ``spawn`` method therefore
cannot reliably pickle functions defined inside the Streamlit script after a
hot reload. Keeping the process target in this ordinary module gives it a
stable import path in both the parent and spawned child.
"""


def paddle_worker_process(model_type, ocr_language, task_q, result_q):
    """Load one Paddle model and serve prediction tasks until shutdown."""
    try:
        import os

        from app.core.config import VLLM_MODEL_NAME, VLLM_SERVER_URL

        os.environ["FLAGS_use_eager_mode"] = "1"

        import gc
        import numpy as np
        import paddle

        paddle.disable_static()
        gc.collect()
        try:
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

        if model_type == "vl":
            from paddleocr import PaddleOCRVL

            model = PaddleOCRVL(
                layout_detection_model_name="PP-DocLayoutV3",
                vl_rec_backend="vllm-server",
                vl_rec_server_url=VLLM_SERVER_URL,
                vl_rec_model_name=VLLM_MODEL_NAME,
                vl_rec_max_concurrency=4,
            )
        else:
            from paddleocr import PPStructureV3

            recognition_model = (
                f"{ocr_language}_PP-OCRv4_mobile_rec"
                if ocr_language == "en"
                else f"{ocr_language}_PP-OCRv4_rec"
            )
            model = PPStructureV3(
                layout_detection_model_name="PP-DocLayout-L",
                text_recognition_model_name=recognition_model,
                device="gpu",
                use_table_recognition=True,
                use_doc_orientation_classify=False,
                use_region_detection=True,
                use_doc_unwarping=False,
            )
        result_q.put(("ready", None))
    except Exception as error:
        result_q.put(("init_error", f"{type(error).__name__}: {error}"))
        return

    while True:
        try:
            task = task_q.get(timeout=300)
        except Exception:
            break

        if task is None:
            break

        image_bytes, image_shape = task
        try:
            image_bgr = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)
            raw_results = list(
                model.predict(image_bgr, use_queues=True, max_pixels=1003520)
            )
            serialized = [
                result.json if hasattr(result, "json") else result
                for result in raw_results
            ]
            result_q.put(("ok", serialized))
        except Exception as error:
            result_q.put(("error", f"{type(error).__name__}: {error}"))
            break
