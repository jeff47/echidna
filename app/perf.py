from __future__ import annotations

import os


_TRUTHY_VALUES = {"1", "true", "yes", "on"}
PERF_LOGGING_ENABLED = os.getenv("ECHIDNA_PERF_LOG", "").strip().lower() in _TRUTHY_VALUES


def perf_logging_enabled() -> bool:
    return PERF_LOGGING_ENABLED
