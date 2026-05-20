"""JSON serialisation helpers for Evo2 NIM tool responses.

Tool functions build their own dicts inline in `server.py` — schema lives
in each tool's docstring. This module exists only for the things that need
real transformation (no formatters left that are pure dict-wrapping) plus
the shared `dump` helper.
"""

from __future__ import annotations

import json
import math


def _json_safe(obj):
    """Fallback for numpy scalars + non-finite floats. Keep numpy import lazy."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if not math.isfinite(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitise_floats(value):
    """Recursively replace non-finite floats with None + coerce numpy scalars.

    Numpy floats with NaN can't pass json.dumps's allow_nan=False even via the
    `default=` fallback (the encoder fails before calling it), so we strip them
    up front along with native Python floats.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Lazy numpy check — keep formatters numpy-optional at import time.
    try:
        import numpy as np

        if isinstance(value, np.floating):
            v = float(value)
            return v if math.isfinite(v) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return [_sanitise_floats(v) for v in value.tolist()]
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, dict):
        return {k: _sanitise_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitise_floats(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitise_floats(v) for v in value]
    return value


def dump(payload) -> str:
    """JSON-serialise `payload`: NaN/Inf → null, numpy scalars → Python natives."""
    return json.dumps(
        _sanitise_floats(payload),
        default=_json_safe,
        allow_nan=False,
        separators=(",", ":"),
    )


def runtime(server_ms: float | None, total_ms: float | None) -> dict:
    """Standard runtime sub-dict every tool includes."""
    return {"server_ms": server_ms, "total_ms": total_ms}
