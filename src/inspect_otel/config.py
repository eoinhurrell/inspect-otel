"""Configuration via environment variables."""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("INSPECT_OTEL_ENABLED", "true").lower() in ("true", "1", "yes")


def capture_content() -> bool:
    return os.environ.get("INSPECT_OTEL_CAPTURE_CONTENT", "false").lower() in ("true", "1", "yes")


def service_name() -> str:
    return os.environ.get("INSPECT_OTEL_SERVICE_NAME", "inspect")


def sample_span_limit() -> int | None:
    val = os.environ.get("INSPECT_OTEL_SAMPLE_SPAN_LIMIT")
    return int(val) if val is not None else None
