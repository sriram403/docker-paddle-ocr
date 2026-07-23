"""Headless progress adapter used by the API processing engine.

The experimentation pipeline reports progress through Streamlit.  The Docker
service runs the same pipeline without a Streamlit script context, so this
module provides the small subset of that interface used by the core engine.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable


logger = logging.getLogger("pipeline_progress")


class _Progress:
    def progress(self, *_args: Any, **_kwargs: Any) -> "_Progress":
        return self

    def empty(self) -> None:
        return None


class _Status(_Progress):
    def __enter__(self) -> "_Status":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class HeadlessStreamlit:
    """Minimal Streamlit-compatible facade for non-UI pipeline execution."""

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}

    def cache_resource(self, func: Callable | None = None, **_kwargs: Any):
        def decorator(inner: Callable) -> Callable:
            return inner

        return decorator(func) if func is not None else decorator

    def progress(self, *_args: Any, **_kwargs: Any) -> _Progress:
        return _Progress()

    def status(self, *_args: Any, **_kwargs: Any) -> _Status:
        return _Status()

    def spinner(self, *_args: Any, **_kwargs: Any):
        return nullcontext()

    def write(self, message: Any = "", *_args: Any, **_kwargs: Any) -> None:
        logger.info("%s", message)

    def info(self, message: Any = "", *_args: Any, **_kwargs: Any) -> None:
        logger.info("%s", message)

    def warning(self, message: Any = "", *_args: Any, **_kwargs: Any) -> None:
        logger.warning("%s", message)

    def error(self, message: Any = "", *_args: Any, **_kwargs: Any) -> None:
        logger.error("%s", message)


st = HeadlessStreamlit()
