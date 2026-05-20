"""Round-trip / shape tests for evo2_nim_mcp.formatters JSON helpers.

The markdown formatters were removed when we switched to JSON-only tool output.
What remains in the module is the `dump` helper (NaN-safe + numpy-aware
json.dumps) and a small `runtime` envelope. Both are exercised here.

Tests that the old `format_*` functions are gone live at the bottom — a
canary against regressions where someone reintroduces a markdown path.
"""

from __future__ import annotations

import json
import math

import numpy as np

from evo2_nim_mcp import formatters


def test_dump_nan_inf_coerced():
    out = json.loads(formatters.dump({"a": float("nan"), "b": float("inf"), "c": 1.5}))
    assert out == {"a": None, "b": None, "c": 1.5}


def test_dump_numpy_scalars():
    out = json.loads(formatters.dump({
        "i": np.int64(42),
        "f": np.float32(2.5),
        "f_nan": np.float64(np.nan),
        "b": np.bool_(False),
        "arr": np.array([0.0, np.nan]),
    }))
    assert out == {"i": 42, "f": 2.5, "f_nan": None, "b": False, "arr": [0.0, None]}


def test_runtime_envelope():
    rt = formatters.runtime(12.5, 30.0)
    assert rt == {"server_ms": 12.5, "total_ms": 30.0}
    rt_none = formatters.runtime(None, None)
    assert rt_none == {"server_ms": None, "total_ms": None}


def test_dump_round_trips_a_realistic_score_payload():
    # Mirrors what server.py's score_variant_at builds inline.
    payload = {
        "variant": {
            "chromosome": "chr1",
            "position": 196716375,
            "ref": "T",
            "alt": "C",
            "assembly": "GRCh38",
        },
        "context": {
            "chromosome": "chr1",
            "start": 196712183,
            "end": 196720375,
            "length": 8192,
        },
        "scores": {
            "score_ref": -0.676672,
            "score_alt": -0.677868,
            "score_delta": -0.001196,
            "reduce_method": "mean",
        },
        "strand_swap_note": None,
        "runtime": formatters.runtime(2300.0, 2310.0),
    }
    s = formatters.dump(payload)
    out = json.loads(s)
    assert out["variant"]["position"] == 196716375
    assert math.isclose(out["scores"]["score_delta"], -0.001196, rel_tol=1e-6)
    assert out["strand_swap_note"] is None
    assert out["runtime"]["server_ms"] == 2300.0


def test_dump_compact_separators():
    """No whitespace between keys/values — keeps payload tight on the wire."""
    s = formatters.dump({"a": 1, "b": 2})
    assert "," in s and " " not in s


def test_invalid_nonfinite_inside_nested_list():
    payload = {"results": [{"score": float("nan")}, {"score": 0.5}]}
    out = json.loads(formatters.dump(payload))
    assert out["results"][0]["score"] is None
    assert out["results"][1]["score"] == 0.5


def test_no_markdown_formatters_left():
    """Canary: the old format_* functions are removed; only dump + runtime remain.

    If anyone reintroduces markdown rendering at this layer, this test fires.
    """
    for attr in (
        "format_score_sequence",
        "format_score_snp",
        "format_score_variant_batch",
        "format_score_splice_region",
        "format_embed_sequence",
        "format_embed_similarity",
        "format_generate_sequence",
        "format_list_checkpoints",
        "format_list_layer_names",
        "format_fetch_variant_context",
        "format_score_variant_at",
        "format_nim_health",
    ):
        assert not hasattr(formatters, attr), f"{attr} should have been removed"
