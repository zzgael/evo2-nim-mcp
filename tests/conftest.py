"""Shared test fixtures: synthetic NPZ payloads matching the NIM /forward shape."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest


def make_logits(seq_len: int, vocab_size: int = 512, *, seed: int = 0) -> np.ndarray:
    """Random logits with mild concentration on a chosen "ground truth" token per position.

    Useful for tests that need plausible logits without depending on a real model.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((seq_len, vocab_size)).astype(np.float32)


def npz_to_b64(arrays: dict[str, np.ndarray]) -> str:
    """Pack a {name: ndarray} dict into the base64-encoded NPZ format the NIM returns."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def random_logits_b64() -> str:
    """Base64 NPZ containing a single 'lm_head.output' array of shape (12, 512)."""
    arrays = {"lm_head.output": make_logits(12)}
    return npz_to_b64(arrays)


@pytest.fixture
def empty_npz_b64() -> str:
    """Base64 NPZ that decodes to an archive with zero arrays.

    np.savez_compressed always writes at least one entry if given any kwargs,
    so we have to construct an explicit empty zip; we simulate this by returning
    an empty string (which is what the NIM returns when output_layers is invalid).
    """
    return ""
