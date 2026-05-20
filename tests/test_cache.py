"""Tests for the Evo2 result cache.

We don't mock the NIM here — these tests exercise the cache module in
isolation. End-to-end cache-hit assertions live in test_server.py.
"""

from __future__ import annotations

import json
import os

import pytest

from evo2_nim_mcp import cache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Each test gets a fresh empty cache at a temp path."""
    monkeypatch.setenv("EVO2_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("EVO2_CACHE_DISABLED", raising=False)
    monkeypatch.setenv("NIM_VARIANT", "40b")
    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


def test_put_and_get_round_trip():
    cache.put("score_sequence", ("hash_a", "mean"), {"score": -1.5, "server_ms": 1200})
    got = cache.get("score_sequence", ("hash_a", "mean"))
    assert got == {"score": -1.5, "server_ms": 1200}


def test_miss_returns_none():
    assert cache.get("score_sequence", ("never_written", "mean")) is None


def test_variant_scoping():
    cache.put("score_sequence", ("seq1", "mean"), {"score": 1.0})
    # Switch model variant — same key should miss
    os.environ["NIM_VARIANT"] = "7b"
    assert cache.get("score_sequence", ("seq1", "mean")) is None
    os.environ["NIM_VARIANT"] = "40b"
    assert cache.get("score_sequence", ("seq1", "mean")) == {"score": 1.0}


def test_disable_blocks_reads_and_writes(monkeypatch):
    cache.put("score_sequence", ("seq1", "mean"), {"score": 1.0})
    monkeypatch.setenv("EVO2_CACHE_DISABLED", "1")
    # Read returns None even though the row exists
    assert cache.get("score_sequence", ("seq1", "mean")) is None
    # Write is silently dropped
    cache.put("score_sequence", ("seq2", "mean"), {"score": 2.0})
    monkeypatch.delenv("EVO2_CACHE_DISABLED")
    assert cache.get("score_sequence", ("seq2", "mean")) is None


def test_keys_are_deterministic():
    k1 = cache.score_sequence_key("ACGTACGT", "mean")
    k2 = cache.score_sequence_key("ACGTACGT", "mean")
    assert k1 == k2
    k3 = cache.score_sequence_key("ACGTACGT", "sum")
    assert k1 != k3
    k4 = cache.score_sequence_key("ACGTACGG", "mean")
    assert k1 != k4


def test_embed_similarity_key_is_order_independent():
    a = "ACGTACGT"
    b = "TGCATGCA"
    assert cache.embed_similarity_key(a, b, "L20") == cache.embed_similarity_key(b, a, "L20")
    # But different layer differs
    assert cache.embed_similarity_key(a, b, "L20") != cache.embed_similarity_key(a, b, "L30")


def test_stats_counts_rows():
    cache.put("score_sequence", ("a", "mean"), {"score": 1})
    cache.put("score_sequence", ("b", "mean"), {"score": 2})
    cache.put("embed_sequence", ("c", "L20"), {"shape": [1, 8192]})
    s = cache.stats()
    assert s["cache_version"] == cache.CACHE_VERSION
    ns_counts = {(n["namespace"], n["variant"]): n["rows"] for n in s["namespaces"]}
    assert ns_counts[("score_sequence", "40b")] == 2
    assert ns_counts[("embed_sequence", "40b")] == 1


def test_version_bump_invalidates(monkeypatch):
    cache.put("score_sequence", ("seq1", "mean"), {"score": 1.0})
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    # New version → fresh namespace; old row is invisible without manual lookup
    assert cache.get("score_sequence", ("seq1", "mean")) is None


def test_non_finite_floats_handled():
    # NaN/Inf would crash json.dumps unless we pre-coerce. The cache module
    # currently uses allow_nan default — verify we surface that contract
    # cleanly so callers know to sanitise before put.
    with pytest.raises(ValueError):
        cache.put("score_sequence", ("seq", "mean"), {"score": float("nan")})
