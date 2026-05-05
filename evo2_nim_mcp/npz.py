"""Decode the NIM `/forward` response into numpy arrays.

The NIM returns:
    {"data": "<base64-encoded NPZ archive>", "elapsed_ms": int}

The NPZ archive contains one array per requested `output_layers` entry, keyed
by the layer name. Each array has shape `(seq_len, hidden_dim)` for hidden
states or `(seq_len, vocab_size)` for the LM head logits.
"""

from __future__ import annotations

import base64
import io

import numpy as np


class NpzDecodeError(Exception):
    """Raised when the base64 NPZ payload cannot be decoded."""


def decode_forward_response(data_b64: str) -> dict[str, np.ndarray]:
    """Decode the base64 NPZ payload returned by `/biology/arc/evo2/forward`.

    Returns a mapping {layer_name: ndarray}. Raises NpzDecodeError on
    malformed input — the caller should surface this with a clear message
    naming the requested layers, since most commonly the NIM returned an
    empty payload because the layer name was wrong.
    """
    if not data_b64:
        raise NpzDecodeError(
            "NIM /forward returned an empty `data` field. "
            "Most often this means one of the requested `output_layers` does not exist for this checkpoint. "
            "Call `list_layer_names` to see valid layer names."
        )

    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise NpzDecodeError(f"NIM /forward returned non-base64 data: {exc}") from exc

    try:
        npz = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise NpzDecodeError(
            f"NIM /forward payload is not a valid NPZ archive: {type(exc).__name__}: {exc}"
        ) from exc

    # Materialize each array eagerly so any decode-time errors (e.g. pickled
    # object arrays under allow_pickle=False) are caught and re-raised as
    # NpzDecodeError instead of bubbling up as raw ValueError.
    arrays: dict[str, np.ndarray] = {}
    try:
        for name in npz.files:
            arrays[name] = npz[name]
    except Exception as exc:
        raise NpzDecodeError(
            f"NIM /forward NPZ archive contains unreadable array {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not arrays:
        raise NpzDecodeError(
            "NIM /forward returned an NPZ archive with zero arrays. "
            "Check that the requested `output_layers` are valid for this checkpoint."
        )
    return arrays
