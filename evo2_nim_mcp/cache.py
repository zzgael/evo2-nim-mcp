"""SQLite-backed cache for Evo2 NIM results.

Evo2 inference is deterministic — same (sequence, reduce_method, model
variant) always produces the same score / embedding stats. That makes
caching a free correctness win: every cache hit skips an Arkane GPU call.

Why this matters in practice: research sessions iterate. The LLM writes
a scoring script, smoke-tests it on a small fixture, then runs it on the
full manifest, then re-runs with tweaked params. Many of those calls
score the SAME sequence we already scored an hour ago. Without a cache,
each one burns ~17 s of GPU.

Design choices:
- **SQLite at $HOME/.cache/evo2-nim-mcp/cache.sqlite** — single file,
  zero-config, survives MCP process restarts. WAL mode so concurrent
  MCP processes (one per tool call) don't lock each other out.
- **Hashed keys** — we sha256 the sequence + params tuple, not store
  the sequence in the row. Keeps the DB tiny even with multi-MB seqs.
- **Variant-scoped** — `NIM_VARIANT` is part of the key. If someone
  redeploys with `evo2_7b` instead of `evo2_40b`, the cache won't
  return stale 40b scores under 7b queries.
- **Version-scoped** — `CACHE_VERSION` constant bumps invalidate the
  whole cache. Bump it on any semantic change (new reduce method,
  fixed bug in scoring, etc.).
- **Opt-out**: `EVO2_CACHE_DISABLED=1` disables cache reads + writes
  globally without removing the call sites. Useful for benchmarking
  or correctness diffs.

Cache values are JSON-serialised dicts. The contract per namespace is
documented at each `get_*`/`put_*` helper below.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

CACHE_VERSION = 1

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _cache_path() -> Path:
    override = os.environ.get("EVO2_CACHE_PATH")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "evo2-nim-mcp" / "cache.sqlite"


def disabled() -> bool:
    return os.environ.get("EVO2_CACHE_DISABLED", "").lower() in ("1", "true", "yes")


def _variant_key() -> str:
    return os.environ.get("NIM_VARIANT", "40b").lower()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(p), check_same_thread=False, isolation_level=None)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS kv (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                variant TEXT NOT NULL,
                cache_version INTEGER NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (namespace, key, variant, cache_version)
            )"""
        )
    return _conn


def _hash_key(parts: tuple) -> str:
    """Deterministic hash of a tuple of strings / numbers."""
    encoded = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seq_hash(sequence: str) -> str:
    """Separate sequence hash so very long seqs don't bloat the key tuple."""
    return hashlib.sha256(sequence.encode()).hexdigest()


def get(namespace: str, key_parts: tuple) -> Any | None:
    if disabled():
        return None
    key = _hash_key(key_parts)
    with _lock:
        row = _get_conn().execute(
            "SELECT value FROM kv WHERE namespace=? AND key=? AND variant=? AND cache_version=?",
            (namespace, key, _variant_key(), CACHE_VERSION),
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None


def put(namespace: str, key_parts: tuple, value: Any) -> None:
    if disabled():
        return
    key = _hash_key(key_parts)
    payload = json.dumps(value, allow_nan=False)
    with _lock:
        _get_conn().execute(
            "INSERT OR REPLACE INTO kv (namespace, key, variant, cache_version, value, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (namespace, key, _variant_key(), CACHE_VERSION, payload, time.time()),
        )


# ---------------------------------------------------------------------------
# Per-tool helpers — keep the cache key construction next to its consumer so
# server.py doesn't have to know the hash composition.
# ---------------------------------------------------------------------------


def score_sequence_key(sequence: str, reduce_method: str) -> tuple:
    return (_seq_hash(sequence), reduce_method)


def score_snp_key(
    sequence: str, alternative_allele: str, position: int | None, reduce_method: str
) -> tuple:
    return (_seq_hash(sequence), alternative_allele, position, reduce_method)


def score_splice_region_key(
    sequence: str,
    splice_position: int,
    reference_dinucleotide: str,
    alternative_dinucleotide: str,
    reduce_method: str,
) -> tuple:
    return (
        _seq_hash(sequence),
        splice_position,
        reference_dinucleotide,
        alternative_dinucleotide,
        reduce_method,
    )


def score_variant_at_key(
    chromosome: str,
    position: int,
    ref_base: str,
    alt_base: str,
    window_size: int,
    species: str,
    assembly: str,
    reduce_method: str,
) -> tuple:
    # Note: cache is keyed on the LLM-side request, not the Ensembl-fetched
    # window. If the chromosome reference changes for the SAME assembly, the
    # cache could return stale results. In practice the human assemblies are
    # frozen at the release version, so this is safe. Bump CACHE_VERSION if
    # we ever support hot-swapping mirrors.
    return (chromosome, position, ref_base, alt_base, window_size, species, assembly, reduce_method)


def embed_sequence_key(sequence: str, layer_name: str) -> tuple:
    return (_seq_hash(sequence), layer_name)


def embed_similarity_key(seq_a: str, seq_b: str, layer_name: str) -> tuple:
    # Order-independent — cos(a, b) == cos(b, a). Hash sorted to dedupe.
    return tuple(sorted([_seq_hash(seq_a), _seq_hash(seq_b)])) + (layer_name,)


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


def stats() -> dict:
    """Return cache size + per-namespace counts for diagnostics."""
    with _lock:
        conn = _get_conn()
        path = _cache_path()
        size_bytes = path.stat().st_size if path.exists() else 0
        rows = conn.execute(
            "SELECT namespace, variant, COUNT(*) FROM kv WHERE cache_version=? GROUP BY namespace, variant",
            (CACHE_VERSION,),
        ).fetchall()
    return {
        "path": str(_cache_path()),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "cache_version": CACHE_VERSION,
        "namespaces": [
            {"namespace": ns, "variant": v, "rows": n} for ns, v, n in rows
        ],
    }


def reset_for_tests() -> None:
    """Drop the in-memory connection and the on-disk file. Tests only."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        p = _cache_path()
        if p.exists():
            p.unlink()
        # WAL/SHM siblings
        for suffix in ("-wal", "-shm"):
            sibling = p.with_name(p.name + suffix)
            if sibling.exists():
                sibling.unlink()
