"""FastMCP server exposing the Evo2 NIM as nine LLM-optimized tools.

The server speaks MCP over stdio — pipe it into an MCP host (Claude Desktop,
Cursor, GPT Workbench, the MCP inspector). At startup it constructs a single
shared `NimClient` from environment variables; every tool call routes through
that client.

The tool docstrings are deliberately rich: USE WHEN / DO NOT USE WHEN /
INTERPRETATION sections give the LLM enough context to pick the right tool
without trial and error.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from fastmcp import FastMCP

from evo2_nim_mcp import cache, formatters, layer_catalog
from evo2_nim_mcp.client import NimClient, NimError, NimNotReadyError
from evo2_nim_mcp.ensembl import EnsemblClient, EnsemblError, FetchedContext
from evo2_nim_mcp.npz import NpzDecodeError, decode_forward_response
from evo2_nim_mcp.scoring import (
    ScoringError,
    apply_snp,
    log_likelihood_from_logits,
    per_position_log_likelihoods,
)

import base64
import io


def _encode_npz(**arrays: np.ndarray) -> str:
    """Pack named arrays into a base64 NPZ blob for inline transport.

    The LLM decodes in code-interp with:
        import base64, io, numpy as np
        a = np.load(io.BytesIO(base64.b64decode(payload)))

    Compressed; suitable for 80 KB - 50 MB payloads. Beyond that the inline
    cost dominates the LLM context and the caller should chunk.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Hard cap: payload over this size after base64 is refused at the source — the
# LLM context window can't hold it cleanly and the savings vs. chunking are nil.
_MAX_INLINE_NPZ_BYTES = 60 * 1024 * 1024  # 60 MB base64 (≈45 MB raw, well within 200K-token windows)

mcp = FastMCP("Evo2 NIM")


def _err(exc_or_message, **details) -> str:
    """JSON error envelope returned by every tool on failure.

    Shape: {"error": {"type": str, "message": str, ...details}}.
    Keeps the result a parseable JSON string so the LLM only needs one parse path.
    """
    if isinstance(exc_or_message, BaseException):
        payload = {"type": type(exc_or_message).__name__, "message": str(exc_or_message)}
    else:
        payload = {"type": "ToolError", "message": str(exc_or_message)}
    if details:
        payload.update(details)
    return formatters.dump({"error": payload})


# Single shared clients, lazily constructed on first call.
_client: NimClient | None = None
_ensembl: EnsemblClient | None = None


def _get_client() -> NimClient:
    global _client
    if _client is None:
        _client = NimClient.from_env()
    return _client


def _get_ensembl() -> EnsemblClient:
    global _ensembl
    if _ensembl is None:
        _ensembl = EnsemblClient.from_env()
    return _ensembl


def _checkpoint_name() -> str:
    """Best-effort identification of the loaded checkpoint.

    The NIM does not expose a "what model is loaded" endpoint, so we infer
    from the `NIM_VARIANT` env var (which is what configures the container
    at deploy time). Defaults to `evo2_40b` per NIM's own default.
    """
    variant = os.environ.get("NIM_VARIANT", "40b").lower()
    return f"evo2_{variant}"


async def _forward_logits(client: NimClient, sequence: str) -> tuple[np.ndarray, float]:
    """Call /forward with the LM head layer and return (logits, server_elapsed_ms).

    NIM returns logits as a 3D tensor `(seq_len, 1, vocab)`. We squeeze the batch
    dim down to `(seq_len, vocab)` so the scoring functions can consume it.
    """
    response = await client.forward(
        {"sequence": sequence, "output_layers": [layer_catalog.LM_HEAD_LAYER]}
    )
    arrays = decode_forward_response(response.get("data", ""))
    key = layer_catalog.response_key(layer_catalog.LM_HEAD_LAYER)
    if key not in arrays:
        raise NimError(
            f"NIM /forward response did not contain key {key!r}. "
            f"Available: {list(arrays)}. The layer name in "
            "`evo2_nim_mcp.layer_catalog` may be incorrect for this NIM "
            "version — check `list_layer_names` and adjust if needed."
        )
    logits = arrays[key]
    # NIM returns (seq_len, batch=1, vocab); squeeze the batch dim.
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(axis=1)
    return logits, float(response.get("elapsed_ms", 0.0))


# ----------------------------------------------------------------------
# Tool 1 — score_sequence
# ----------------------------------------------------------------------


@mcp.tool()
async def score_sequence(
    sequence: str,
    reduce_method: str = "mean",
    return_per_position: bool = False,
) -> str:
    """Compute the log-likelihood of a DNA sequence under Evo2.

    USE THIS WHEN:
    - User wants to rank candidate sequences by biological plausibility
    - User wants a single scalar score for a whole sequence
    - User wants per-base log-likelihood for plotting / windowed analysis
      (set `return_per_position=True`)

    DO NOT USE WHEN:
    - User has a specific point mutation → prefer `score_snp` (gives a delta)
    - User has multiple variants to score → prefer `score_variant_batch`
    - User has a splice variant → prefer `score_splice_region`

    PARAMETERS:
    - `sequence`: DNA, length ≥ 2, IUPAC alphabet (A, C, G, T, N — upper or lowercase).
    - `reduce_method`: "mean" (default; robust to length) or "sum" (totals all positions).
    - `return_per_position`: when True, also return the per-position log-likelihood
      array as inline base64 NPZ for code-interp analysis. ~5 bytes per position
      on the wire (NPZ-compressed float32), so a 20 kb sequence is ~100 KB base64.

    INTERPRETATION:
    - Higher (less negative) score = more plausible under the model.
    - A score near `log(1/4) ≈ -1.39` ≈ random nucleotides; natural DNA scores higher.

    OUTPUT: JSON. Without `return_per_position`:
      {sequence_length, score, reduce_method, cache, runtime}

    With `return_per_position=True`, the response also includes a `per_position`
    object with `npz_payload` and a decode snippet:
      import base64, io, numpy as np
      a = np.load(io.BytesIO(base64.b64decode(payload)))
      ll = a["ll"]   # shape (seq_len - 1,), float32, log-prob of token i+1 given 1..i
    """
    t0 = time.monotonic()
    cache_key = cache.score_sequence_key(sequence, reduce_method)

    # Fast path: scalar cache only. If the caller asks for per-position data
    # AND we have a scalar-only cached entry, we still have to call NIM to get
    # the array. (Could split cache namespaces to also memoise the array, but
    # that ~doubles cache size on every score; skip for now.)
    if not return_per_position:
        cached = cache.get("score_sequence", cache_key)
        if cached is not None:
            return formatters.dump({
                "sequence_length": len(sequence),
                "score": cached["score"],
                "reduce_method": reduce_method,
                "cache": {"hit": True, "saved_server_ms": cached.get("server_ms")},
                "runtime": formatters.runtime(0, (time.monotonic() - t0) * 1000.0),
            })

    try:
        logits, server_ms = await _forward_logits(_get_client(), sequence)
        score = log_likelihood_from_logits(logits, sequence, reduce_method=reduce_method)
    except (NimError, NimNotReadyError, NpzDecodeError, ScoringError) as exc:
        return _err(exc)

    cache.put(
        "score_sequence",
        cache_key,
        {"score": score, "server_ms": server_ms},
    )

    payload: dict = {
        "sequence_length": len(sequence),
        "score": score,
        "reduce_method": reduce_method,
        "cache": {"hit": False},
        "runtime": formatters.runtime(server_ms, (time.monotonic() - t0) * 1000.0),
    }
    if return_per_position:
        try:
            ll_arr = per_position_log_likelihoods(logits, sequence)
        except ScoringError as exc:
            return _err(exc)
        npz = _encode_npz(ll=ll_arr)
        payload["per_position"] = {
            "shape": list(ll_arr.shape),
            "dtype": "float32",
            "npz_payload": npz,
            "decode": (
                "import base64, io, numpy as np; "
                "a = np.load(io.BytesIO(base64.b64decode(payload))); "
                "ll = a['ll']  # shape (seq_len - 1,)"
            ),
        }
    return formatters.dump(payload)


# ----------------------------------------------------------------------
# Tool 2 — score_snp
# ----------------------------------------------------------------------


@mcp.tool()
async def score_snp(
    sequence: str,
    alternative_allele: str,
    reduce_method: str = "mean",
    position: int | None = None,
) -> str:
    """Score a single nucleotide variant by computing the log-likelihood of the
    reference vs the mutated sequence under Evo2 and returning the delta.

    USE THIS WHEN:
    - User asks about pathogenicity / effect of a specific point mutation
    - User has a sequence + a single alternate base to substitute
    - User wants a quantitative score for variant ranking

    DO NOT USE WHEN:
    - User has multiple variants → prefer `score_variant_batch`
    - The variant is at a splice junction → prefer `score_splice_region`
    - User wants whole-sequence plausibility → prefer `score_sequence`

    PARAMETERS:
    - `sequence`: DNA reference, length ≥ 3, IUPAC alphabet. The mutation is
      applied at the center position by default.
    - `alternative_allele`: single nucleotide (A, C, G, T, N) replacing the
      reference base at `position`.
    - `reduce_method`: "mean" (default, robust) or "sum" (sensitive to length).
    - `position`: 0-based index of the base to mutate. Defaults to the center
      of the sequence (matches the upstream evo2-mcp convention).

    OUTPUT:
    - `score_delta = LL(alt) − LL(ref)` (per-position log-likelihood, averaged
      with `reduce_method='mean'`).
    - Sign convention: negative delta = mutated window has lower likelihood
      than the reference under the model. The magnitude carries no calibrated
      mapping to clinical pathogenicity — it's a model-internal signal.

    Returns markdown with LL(ref), LL(alt), delta, window length, and the
    raw sequences. The caller is responsible for interpretation.
    """
    t0 = time.monotonic()
    cached = cache.get(
        "score_snp",
        cache.score_snp_key(sequence, alternative_allele, position, reduce_method),
    )
    if cached is not None:
        return formatters.dump({
            "position": cached["position"],
            "reference_allele": cached["reference_allele"],
            "alternative_allele": alternative_allele,
            "score_ref": cached["score_ref"],
            "score_alt": cached["score_alt"],
            "score_delta": cached["score_alt"] - cached["score_ref"],
            "reduce_method": reduce_method,
            "window_length": len(sequence),
            "cache": {"hit": True, "saved_server_ms": cached.get("server_ms")},
            "runtime": formatters.runtime(0, (time.monotonic() - t0) * 1000.0),
        })
    try:
        mutated, pos = apply_snp(sequence, alternative_allele, position=position)
        client = _get_client()
        ref_logits, ref_server_ms = await _forward_logits(client, sequence)
        alt_logits, alt_server_ms = await _forward_logits(client, mutated)
        score_ref = log_likelihood_from_logits(ref_logits, sequence, reduce_method=reduce_method)
        score_alt = log_likelihood_from_logits(alt_logits, mutated, reduce_method=reduce_method)
    except (NimError, NimNotReadyError, NpzDecodeError, ScoringError) as exc:
        return _err(exc)

    cache.put(
        "score_snp",
        cache.score_snp_key(sequence, alternative_allele, position, reduce_method),
        {
            "position": pos,
            "reference_allele": sequence[pos],
            "score_ref": score_ref,
            "score_alt": score_alt,
            "server_ms": ref_server_ms + alt_server_ms,
        },
    )
    return formatters.dump({
        "position": pos,
        "reference_allele": sequence[pos],
        "alternative_allele": alternative_allele,
        "score_ref": score_ref,
        "score_alt": score_alt,
        "score_delta": score_alt - score_ref,
        "reduce_method": reduce_method,
        "window_length": len(sequence),
        "cache": {"hit": False},
        "runtime": formatters.runtime(ref_server_ms + alt_server_ms, (time.monotonic() - t0) * 1000.0),
    })


# ----------------------------------------------------------------------
# Tool 3 — score_variant_batch
# ----------------------------------------------------------------------


@mcp.tool()
async def score_variant_batch(
    variants: list[dict[str, Any]],
    reduce_method: str = "mean",
) -> str:
    """Batch-score multiple SNPs against the same or different sequences.

    USE THIS WHEN:
    - User has a cohort or VCF-like list of point variants to triage
    - User wants a ranked table of pathogenicity scores for many variants
    - You're orchestrating a downstream nephro-variant-triage workflow

    DO NOT USE WHEN:
    - User has just one variant → use `score_snp` directly
    - User wants splice-aware scoring → use `score_splice_region` per variant

    PARAMETERS:
    - `variants`: list of dicts. Each dict must have keys:
        - `sequence` (str): DNA reference, length ≥ 3
        - `alternative_allele` (str): A/C/G/T/N
        - `position` (int, optional): 0-based, defaults to center of sequence
        - `id` (str, optional): caller-provided identifier (echoed in output)
    - `reduce_method`: "mean" (default) or "sum"

    OUTPUT:
    - Markdown table sorted in input order, one row per variant, with
      LL(ref), LL(alt), and `score_delta`.
    - Failures (bad inputs, NIM errors) are reported per-row without aborting
      the whole batch.
    - No clinical / pathogenicity labels are attached — the caller does
      interpretation.

    Returns markdown with summary statistics and per-variant table.
    """
    t0 = time.monotonic()
    client = _get_client()
    results: list[dict[str, Any]] = []
    server_ms_total = 0.0
    n_cache_hits = 0
    for v in variants:
        try:
            seq = v["sequence"]
            alt = v["alternative_allele"]
            pos_in = v.get("position")
            key = cache.score_snp_key(seq, alt, pos_in, reduce_method)
            cached = cache.get("score_snp", key)
            if cached is not None:
                n_cache_hits += 1
                results.append({
                    "id": v.get("id"),
                    "position": cached["position"],
                    "reference_allele": cached["reference_allele"],
                    "alternative_allele": alt,
                    "score_ref": cached["score_ref"],
                    "score_alt": cached["score_alt"],
                    "score_delta": cached["score_alt"] - cached["score_ref"],
                    "from_cache": True,
                })
                continue
            mutated, position = apply_snp(seq, alt, position=pos_in)
            ref_logits, r_ms = await _forward_logits(client, seq)
            alt_logits, a_ms = await _forward_logits(client, mutated)
            score_ref = log_likelihood_from_logits(ref_logits, seq, reduce_method=reduce_method)
            score_alt = log_likelihood_from_logits(alt_logits, mutated, reduce_method=reduce_method)
            server_ms_total += r_ms + a_ms
            cache.put(
                "score_snp",
                key,
                {
                    "position": position,
                    "reference_allele": seq[position],
                    "score_ref": score_ref,
                    "score_alt": score_alt,
                    "server_ms": r_ms + a_ms,
                },
            )
            results.append(
                {
                    "id": v.get("id"),
                    "position": position,
                    "reference_allele": seq[position],
                    "alternative_allele": alt,
                    "score_ref": score_ref,
                    "score_alt": score_alt,
                    "score_delta": score_alt - score_ref,
                    "from_cache": False,
                }
            )
        except (NimError, NimNotReadyError, NpzDecodeError, ScoringError, KeyError) as exc:
            results.append({"id": v.get("id"), "error": str(exc)})

    n_failures = sum(1 for r in results if "error" in r)
    return formatters.dump({
        "n": len(results),
        "n_success": len(results) - n_failures,
        "n_failures": n_failures,
        "n_cache_hits": n_cache_hits,
        "reduce_method": reduce_method,
        "results": results,
        "runtime": formatters.runtime(server_ms_total, (time.monotonic() - t0) * 1000.0),
    })


# ----------------------------------------------------------------------
# Tool 4 — score_splice_region
# ----------------------------------------------------------------------


@mcp.tool()
async def score_splice_region(
    sequence: str,
    splice_position: int,
    reference_dinucleotide: str,
    alternative_dinucleotide: str,
    reduce_method: str = "mean",
) -> str:
    """Score a splice site change by computing the likelihood delta of the
    sequence with the splice-site dinucleotide replaced.

    USE THIS WHEN:
    - User has a variant that affects a canonical splice donor (GT) or
      acceptor (AG) at the boundary of an intron
    - User wants splice-aware scoring rather than single-base scoring
    - VEP / AlphaGenome flagged a splice variant and you want Evo2's view

    DO NOT USE WHEN:
    - The variant is in the coding region away from splice boundaries → use `score_snp`
    - User has a list of variants → prefer `score_variant_batch` per-variant

    PARAMETERS:
    - `sequence`: full DNA region containing the splice site
    - `splice_position`: 0-based index of the FIRST nucleotide of the splice
      dinucleotide (`reference_dinucleotide` occupies positions
      `splice_position` and `splice_position + 1`)
    - `reference_dinucleotide`: the wild-type dinucleotide (typically "GT" or "AG")
    - `alternative_dinucleotide`: the variant dinucleotide (e.g. "GC" or "AT")
    - `reduce_method`: "mean" (default) or "sum"

    OUTPUT:
    - LL(ref), LL(alt), `score_delta`, and whether the reference dinucleotide
      is a canonical donor (GT) or acceptor (AG) motif.
    - For splice-effect prediction proper, prefer SpliceAI or AlphaGenome's
      splice track — Evo2 is a generic DNA language model, not a splice
      classifier.

    Returns markdown with reference / mutated scores, delta, and motif type.
    """
    t0 = time.monotonic()
    cached = cache.get(
        "score_splice_region",
        cache.score_splice_region_key(
            sequence,
            splice_position,
            reference_dinucleotide,
            alternative_dinucleotide,
            reduce_method,
        ),
    )
    if cached is not None:
        return formatters.dump({
            "splice_position": splice_position,
            "reference_dinucleotide": reference_dinucleotide,
            "alternative_dinucleotide": alternative_dinucleotide,
            "score_ref": cached["score_ref"],
            "score_alt": cached["score_alt"],
            "score_delta": cached["score_alt"] - cached["score_ref"],
            "canonical": cached["canonical"],
            "region_length": len(sequence),
            "cache": {"hit": True, "saved_server_ms": cached.get("server_ms")},
            "runtime": formatters.runtime(0, (time.monotonic() - t0) * 1000.0),
        })
    try:
        if len(reference_dinucleotide) != 2 or len(alternative_dinucleotide) != 2:
            raise ScoringError(
                "reference_dinucleotide and alternative_dinucleotide must each be exactly 2 nucleotides."
            )
        if not 0 <= splice_position <= len(sequence) - 2:
            raise ScoringError(
                f"splice_position {splice_position} is out of range for sequence of length {len(sequence)}."
            )
        if sequence[splice_position : splice_position + 2].upper() != reference_dinucleotide.upper():
            raise ScoringError(
                f"reference_dinucleotide {reference_dinucleotide!r} does not match the sequence at "
                f"positions {splice_position}-{splice_position + 1} "
                f"({sequence[splice_position : splice_position + 2]!r})."
            )

        mutated = (
            sequence[:splice_position]
            + alternative_dinucleotide
            + sequence[splice_position + 2 :]
        )
        canonical = reference_dinucleotide.upper() in {"GT", "AG"}

        client = _get_client()
        ref_logits, r_ms = await _forward_logits(client, sequence)
        alt_logits, a_ms = await _forward_logits(client, mutated)
        score_ref = log_likelihood_from_logits(ref_logits, sequence, reduce_method=reduce_method)
        score_alt = log_likelihood_from_logits(alt_logits, mutated, reduce_method=reduce_method)
    except (NimError, NimNotReadyError, NpzDecodeError, ScoringError) as exc:
        return _err(exc)

    cache.put(
        "score_splice_region",
        cache.score_splice_region_key(
            sequence,
            splice_position,
            reference_dinucleotide,
            alternative_dinucleotide,
            reduce_method,
        ),
        {
            "score_ref": score_ref,
            "score_alt": score_alt,
            "canonical": canonical,
            "server_ms": r_ms + a_ms,
        },
    )
    return formatters.dump({
        "splice_position": splice_position,
        "reference_dinucleotide": reference_dinucleotide,
        "alternative_dinucleotide": alternative_dinucleotide,
        "score_ref": score_ref,
        "score_alt": score_alt,
        "score_delta": score_alt - score_ref,
        "canonical": canonical,
        "region_length": len(sequence),
        "cache": {"hit": False},
        "runtime": formatters.runtime(r_ms + a_ms, (time.monotonic() - t0) * 1000.0),
    })


# ----------------------------------------------------------------------
# Tool 5 — embed_sequence
# ----------------------------------------------------------------------


# Embed-tensor inline limits. `full` mode emits the entire (seq_len, hidden)
# tensor as base64 NPZ; payload ≈ seq_len × hidden × 2 bytes (float16) ×
# ~1.3x base64 overhead. 2 kb at 8192-d float16 ≈ 44 MB base64 → just under
# the 60 MB ceiling. Beyond that, chunking is the right move.
_EMBED_FULL_MAX_SEQ_LEN = 2000


@mcp.tool()
async def embed_sequence(
    sequence: str,
    layer_name: str | None = None,
    return_mode: str = "stats",
) -> str:
    """Extract embeddings for a DNA sequence at a named hidden layer.

    USE THIS WHEN:
    - User wants vector representations for similarity search, clustering,
      or downstream classification.
    - User wants the raw tensor for code-interp analysis (set
      `return_mode='pooled'` for a single 8192-d vector per sequence, or
      `return_mode='full'` for the full (seq_len, hidden) tensor).

    DO NOT USE WHEN:
    - User wants a single scalar likelihood score → use `score_sequence`
    - User wants pairwise similarity only → use `embed_similarity`

    PARAMETERS:
    - `sequence`: DNA, IUPAC alphabet, length ≥ 1.
    - `layer_name`: which hidden layer to extract. None = checkpoint default
      (e.g. `decoder.layers.20.mlp` for the 40b).
    - `return_mode`:
        - `"stats"` (default): shape + L2-norm summary only. Cheap; for
          smell-test / "is this OOD?".
        - `"pooled"`: also include the mean-pooled 8192-d vector as inline
          base64 NPZ. ~32 KB on the wire. Use this for cross-sequence
          similarity / clustering.
        - `"full"`: also include the full (seq_len, hidden_dim) float16
          tensor as inline base64 NPZ. Refused above seq_len = 2000
          (payload would exceed 60 MB context budget). Use for per-position
          analysis or chunk the sequence yourself.

    OUTPUT: JSON. With `return_mode='stats'`:
      {layer_name, sequence_length, embedding_shape, norm_mean, norm_std,
       sample_values_first_column, cache, runtime}

    With `return_mode='pooled'`, the response also includes:
      {"pooled": {"shape": [hidden_dim], "dtype": "float32",
                  "npz_payload": "...", "decode": "..."}}

    With `return_mode='full'`, the response also includes:
      {"full": {"shape": [seq_len, hidden_dim], "dtype": "float16",
                "npz_payload": "...", "decode": "..."}}
    """
    if return_mode not in {"stats", "pooled", "full"}:
        return _err(f"return_mode must be 'stats' | 'pooled' | 'full' (got {return_mode!r}).")
    if return_mode == "full" and len(sequence) > _EMBED_FULL_MAX_SEQ_LEN:
        return _err(
            f"return_mode='full' refused: sequence length {len(sequence)} exceeds the "
            f"{_EMBED_FULL_MAX_SEQ_LEN}-position inline cap. Chunk the sequence and "
            "embed each chunk separately, or use return_mode='pooled' for a single vector."
        )

    t0 = time.monotonic()
    checkpoint = _checkpoint_name()
    layer = layer_name or layer_catalog.default_embedding_layer(checkpoint)
    cache_key = cache.embed_sequence_key(sequence, layer)

    # Stats-only fast path: scalar-cached entries cover this. The 'pooled' and
    # 'full' modes need the actual tensor, which the cache doesn't store, so we
    # bypass the cache for those.
    if return_mode == "stats":
        cached = cache.get("embed_sequence", cache_key)
        if cached is not None:
            return formatters.dump({
                "layer_name": layer,
                "sequence_length": len(sequence),
                "embedding_shape": cached["embedding_shape"],
                "norm_mean": cached["norm_mean"],
                "norm_std": cached["norm_std"],
                "sample_values_first_column": cached.get("sample_values_first_column"),
                "cache": {"hit": True, "saved_server_ms": cached.get("server_ms")},
                "runtime": formatters.runtime(0, (time.monotonic() - t0) * 1000.0),
            })

    try:
        client = _get_client()
        response = await client.forward({"sequence": sequence, "output_layers": [layer]})
        arrays = decode_forward_response(response.get("data", ""))
        key = layer_catalog.response_key(layer)
        if key not in arrays:
            return _err(
                f"NIM /forward did not return key {key!r}",
                available_keys=list(arrays),
                hint="Call list_layer_names to see the catalog of supported layer names.",
            )
        embedding = arrays[key]
        # NIM returns (seq_len, batch=1, hidden); squeeze batch to (seq_len, hidden).
        if embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(axis=1)
        norms = np.linalg.norm(embedding.astype(np.float64), axis=-1)
    except (NimError, NimNotReadyError, NpzDecodeError) as exc:
        return _err(exc)

    payload = {
        "layer_name": layer,
        "sequence_length": len(sequence),
        "embedding_shape": list(embedding.shape),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "sample_values_first_column": [
            float(embedding[i, 0]) for i in range(min(5, embedding.shape[0]))
        ],
    }
    cache.put(
        "embed_sequence",
        cache_key,
        {**payload, "server_ms": float(response.get("elapsed_ms", 0.0))},
    )

    out: dict = {
        **payload,
        "cache": {"hit": False},
        "runtime": formatters.runtime(
            float(response.get("elapsed_ms", 0.0)),
            (time.monotonic() - t0) * 1000.0,
        ),
    }

    if return_mode == "pooled":
        pooled = embedding.astype(np.float32).mean(axis=0)
        out["pooled"] = {
            "shape": list(pooled.shape),
            "dtype": "float32",
            "npz_payload": _encode_npz(pooled=pooled),
            "decode": (
                "import base64, io, numpy as np; "
                "a = np.load(io.BytesIO(base64.b64decode(payload))); "
                "v = a['pooled']  # shape (hidden_dim,), float32"
            ),
        }
    elif return_mode == "full":
        # float16 halves payload vs float32 with negligible info loss for embeddings
        emb_fp16 = embedding.astype(np.float16)
        out["full"] = {
            "shape": list(emb_fp16.shape),
            "dtype": "float16",
            "npz_payload": _encode_npz(embedding=emb_fp16),
            "decode": (
                "import base64, io, numpy as np; "
                "a = np.load(io.BytesIO(base64.b64decode(payload))); "
                "emb = a['embedding']  # shape (seq_len, hidden_dim), float16"
            ),
        }
    return formatters.dump(out)


# ----------------------------------------------------------------------
# Tool 5b — embed_similarity (cosine between two sequences)
# ----------------------------------------------------------------------


@mcp.tool()
async def embed_similarity(
    sequence_a: str,
    sequence_b: str,
    layer_name: str | None = None,
) -> str:
    """Compute cosine similarity between embeddings of two DNA sequences.

    USE THIS WHEN:
    - User wants to compare two variants (e.g. wild-type vs mutant flanking
      window) and quantify how much one diverges from the other in
      embedding space
    - User has a cohort of variants and asks "which one is most different
      from WT" — call this pairwise vs the reference
    - You're clustering or ranking sequences by similarity

    DO NOT USE WHEN:
    - User wants raw embeddings → use `embed_sequence`
    - User wants pathogenicity score of a variant → use `score_variant_at`
      or `score_snp` instead — embedding distance ≠ likelihood delta
    - Sequences are different lengths and you need position-wise comparison
      (this tool mean-pools positions, so length-mismatch is tolerated but
      may not reflect what you want)

    PARAMETERS:
    - `sequence_a`, `sequence_b`: DNA, IUPAC alphabet, length ≥ 1.
    - `layer_name`: which hidden layer to embed at. Default depends on
      checkpoint (`decoder.layers.20.mlp` for 40B). Both sequences are
      embedded at the same layer.

    OUTPUT:
    - `cosine_similarity_mean_pool`: mean-pool each embedding over the
      sequence dimension, then cosine of the two pooled vectors. Range
      −1 (anti-parallel) to +1 (identical). Compares overall representation.
    - `cosine_similarity_centered`: only when both sequences have the same
      length — average cosine across position-wise pairs. More sensitive to
      local differences than mean-pool.
    - No published threshold maps cosine values to functional or clinical
      categories. The useful pattern is comparing a panel of variants vs a
      common reference, then ranking by divergence.

    Returns markdown with both metrics, the layer used, and runtime.
    """
    t0 = time.monotonic()
    checkpoint = _checkpoint_name()
    layer = layer_name or layer_catalog.default_embedding_layer(checkpoint)

    cached = cache.get(
        "embed_similarity", cache.embed_similarity_key(sequence_a, sequence_b, layer)
    )
    if cached is not None:
        return formatters.dump({
            "layer_name": layer,
            "sequence_a_length": len(sequence_a),
            "sequence_b_length": len(sequence_b),
            "cosine_similarity_mean_pool": cached["cosine_similarity_mean_pool"],
            "cosine_similarity_centered": cached["cosine_similarity_centered"],
            "cache": {"hit": True, "saved_server_ms": cached.get("server_ms")},
            "runtime": formatters.runtime(0, (time.monotonic() - t0) * 1000.0),
        })

    async def _embed(seq: str) -> tuple[np.ndarray, float]:
        client = _get_client()
        resp = await client.forward({"sequence": seq, "output_layers": [layer]})
        arrays = decode_forward_response(resp.get("data", ""))
        key = layer_catalog.response_key(layer)
        if key not in arrays:
            raise NimError(
                f"NIM /forward did not return key {key!r}. "
                f"Available: {list(arrays)}."
            )
        emb = arrays[key]
        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(axis=1)
        return emb.astype(np.float64), float(resp.get("elapsed_ms", 0.0))

    try:
        emb_a, ms_a = await _embed(sequence_a)
        emb_b, ms_b = await _embed(sequence_b)
    except (NimError, NimNotReadyError, NpzDecodeError) as exc:
        return _err(exc)

    def _cos(u: np.ndarray, v: np.ndarray) -> float:
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu == 0 or nv == 0:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    pooled_a = emb_a.mean(axis=0)
    pooled_b = emb_b.mean(axis=0)
    cos_pool = _cos(pooled_a, pooled_b)

    cos_pos: float | None = None
    if emb_a.shape[0] == emb_b.shape[0]:
        # Position-wise cosine averaged across positions
        per_pos = np.array(
            [_cos(emb_a[i], emb_b[i]) for i in range(emb_a.shape[0])],
            dtype=np.float64,
        )
        cos_pos = float(per_pos.mean())

    cache.put(
        "embed_similarity",
        cache.embed_similarity_key(sequence_a, sequence_b, layer),
        {
            "cosine_similarity_mean_pool": cos_pool,
            "cosine_similarity_centered": cos_pos,
            "server_ms": ms_a + ms_b,
        },
    )
    return formatters.dump({
        "layer_name": layer,
        "sequence_a_length": len(sequence_a),
        "sequence_b_length": len(sequence_b),
        "cosine_similarity_mean_pool": cos_pool,
        "cosine_similarity_centered": cos_pos,
        "cache": {"hit": False},
        "runtime": formatters.runtime(ms_a + ms_b, (time.monotonic() - t0) * 1000.0),
    })


# ----------------------------------------------------------------------
# Tool 6 — generate_sequence
# ----------------------------------------------------------------------


@mcp.tool()
async def generate_sequence(
    prompt: str,
    n_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 4,
    top_p: float = 0.0,
    random_seed: int | None = None,
) -> str:
    """Generate a DNA sequence continuation conditioned on a prompt.

    USE THIS WHEN:
    - User wants Evo2 to design or extend a DNA sequence
    - Workflow D (DNA element design): generate candidate promoters, CRISPR
      guides, regulatory elements
    - User wants stochastic exploration of plausible continuations

    DO NOT USE WHEN:
    - User wants to score an existing sequence → use `score_sequence`
    - User wants embeddings → use `embed_sequence`

    PARAMETERS:
    - `prompt`: starting DNA sequence, IUPAC alphabet
    - `n_tokens`: number of nucleotides to generate (default 100)
    - `temperature`: sampling randomness. <1.0 = more deterministic,
      >1.0 = more diverse. Default 0.7 is a balanced default for nucleotide
      generation.
    - `top_k`: nucleus sampling — keep only the top-k tokens at each step.
      Default 4 (one per nucleotide).
    - `top_p`: nucleus probability cutoff. 0.0 disables; combine with `top_k=0`
      to use top-p sampling instead.
    - `random_seed`: optional integer for reproducibility.

    INTERPRETATION:
    - The generated sequence is a plausible continuation under Evo2;
      it is NOT a verified biological design — downstream validation is required.

    Returns markdown with the generated continuation and runtime.
    """
    t0 = time.monotonic()
    payload: dict[str, Any] = {
        "sequence": prompt,
        "num_tokens": n_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    if random_seed is not None:
        payload["random_seed"] = random_seed

    try:
        response = await _get_client().generate(payload)
    except (NimError, NimNotReadyError) as exc:
        return _err(exc)

    generated = response.get("sequence", "")
    server_ms = float(response.get("elapsed_ms", 0.0))
    return formatters.dump({
        "prompt": prompt,
        "generated": generated,
        "n_tokens": n_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "prompt_length": len(prompt),
        "generated_length": len(generated),
        "runtime": formatters.runtime(server_ms, (time.monotonic() - t0) * 1000.0),
    })


# ----------------------------------------------------------------------
# Tool 7 — list_available_checkpoints
# ----------------------------------------------------------------------


@mcp.tool()
async def list_available_checkpoints() -> str:
    """List the model checkpoints available in this NIM deployment.

    USE THIS WHEN:
    - LLM is unsure which Evo2 model is loaded (40B vs 7B vs other)
    - User asks "which model are we using?"
    - Diagnosing an unexpected scoring or embedding result

    A NIM container exposes exactly one checkpoint at a time (configured via
    the `NIM_VARIANT` env var on the container). To switch checkpoints,
    redeploy the container.
    """
    checkpoint = _checkpoint_name()
    description = (
        "DNA language model from Arc Institute, "
        f"{'40B parameters, 1M context window' if '40b' in checkpoint else '7B parameters, 1M context window' if '7b' in checkpoint else 'configuration unknown'}. "
        "Loaded in this NIM container."
    )
    return formatters.dump({
        "checkpoints": [{"name": checkpoint, "description": description}],
        "note": "One checkpoint per NIM container; configured at deploy time via NIM_VARIANT.",
    })


# ----------------------------------------------------------------------
# Tool 8 — list_layer_names
# ----------------------------------------------------------------------


@mcp.tool()
async def list_layer_names() -> str:
    """List the `output_layers` names accepted by this NIM checkpoint for `/forward`.

    USE THIS WHEN:
    - LLM is about to call `embed_sequence` and needs to know which layer to use
    - LLM gets an "unknown layer" error and needs to discover valid names
    - User asks for a specific layer's purpose (which layers expose what)

    The catalog is curated empirically — names are confirmed by sending
    a small `/forward` request and checking the response. If a name is missing
    here it does NOT mean the NIM rejects it; this is a recommended list, not
    an exhaustive one.
    """
    checkpoint = _checkpoint_name()
    layers = layer_catalog.recommended_for(checkpoint)
    return formatters.dump({"checkpoint": checkpoint, "layers": layers})


# ----------------------------------------------------------------------
# Tool 9 — nim_health
# ----------------------------------------------------------------------


@mcp.tool()
async def cache_stats() -> str:
    """Return Evo2 result-cache size + per-namespace row counts.

    Returns JSON. Shape:
      {path, size_bytes, size_mb, cache_version, disabled,
       namespaces: [{namespace, variant, rows}, ...]}

    The cache deduplicates deterministic Evo2 results (score_sequence,
    score_snp, score_splice_region, score_variant_at, embed_sequence,
    embed_similarity) keyed by sequence/parameters + NIM_VARIANT. Disable
    with EVO2_CACHE_DISABLED=1; override path with EVO2_CACHE_PATH.
    """
    return formatters.dump({**cache.stats(), "disabled": cache.disabled()})


@mcp.tool()
async def nim_health() -> str:
    """Check whether the NIM container is reachable and the model is loaded.

    USE THIS WHEN:
    - Other tool calls fail with connection / timeout errors
    - User asks "is Evo2 up?"
    - Sanity check before kicking off a batch

    Returns markdown with status, the NIM URL, and any extra metadata the
    NIM exposes.
    """
    client = _get_client()
    try:
        body = await client.health()
        return formatters.dump({
            "status": "ready",
            "endpoint": client.base_url,
            "extra": {k: v for k, v in body.items() if k != "status"} or None,
        })
    except NimNotReadyError as exc:
        return formatters.dump({
            "status": "not_ready",
            "endpoint": client.base_url,
            "extra": {"detail": str(exc)},
        })
    except NimError as exc:
        return formatters.dump({
            "status": "unreachable",
            "endpoint": client.base_url,
            "extra": {"detail": str(exc)},
        })


# ----------------------------------------------------------------------
# Tool 10 — fetch_variant_context
# ----------------------------------------------------------------------


@mcp.tool()
async def fetch_variant_context(
    chromosome: str,
    position: int,
    window_size: int = 8192,
    species: str = "human",
    assembly: str = "GRCh38",
) -> str:
    """Fetch reference DNA centred on a genomic coordinate from Ensembl REST.

    USE THIS WHEN:
    - You have a variant by coordinate (chr, pos) and need the flanking DNA
      before calling `score_snp` or `score_sequence`
    - User wants to inspect the wild-type sequence around a variant
    - You're preparing input for `embed_sequence` or `generate_sequence` and
      need a real genomic context rather than a synthetic prompt

    DO NOT USE WHEN:
    - You already have the flanking DNA as a string → call `score_snp` directly
    - You have only an HGVS annotation without coordinates → use `vep_mcp` first
      to resolve the coordinate, then call this tool
    - You need the variant pathogenicity in one shot → call `score_variant_at`
      instead (this tool + score_snp combined)

    PARAMETERS:
    - `chromosome`: chromosome name without prefix (e.g. "16", "X"). `chr` prefix is stripped automatically.
    - `position`: 1-based genomic coordinate of the variant.
    - `window_size`: total length of the flanking window in bp (default 8192,
      matching the Arc Institute BRCA1 zero-shot methodology). Max 10000 unless
      the NIM container has been started with a higher
      `NIM_EVO2_FORWARD_SEQUENCE_LENGTH_LIMIT`.
    - `species`: Ensembl species slug (default "human").
    - `assembly`: "GRCh38" (default) or "GRCh37".

    INTERPRETATION:
    - Returns markdown with the fetched sequence (truncated preview), the
      0-based `center_index` (where `position` maps inside the window), and
      the actual coordinates used. The base at `center_index` is the reference
      allele at `position` on that assembly.

    Returns markdown summary.
    """
    t0 = time.monotonic()
    try:
        ctx = await _get_ensembl().fetch_variant_context(
            chromosome,
            position,
            window_size=window_size,
            species=species,
            assembly=assembly,
        )
    except (EnsemblError, ValueError) as exc:
        return _err(exc)

    return formatters.dump({
        "species": ctx.species,
        "assembly": ctx.assembly,
        "chromosome": ctx.chromosome,
        "start": ctx.start,
        "end": ctx.end,
        "length": len(ctx.sequence),
        "center_index": ctx.center_index,
        "reference_base_at_center": ctx.sequence[ctx.center_index],
        "sequence": ctx.sequence,
        "runtime": formatters.runtime(None, (time.monotonic() - t0) * 1000.0),
    })


# ----------------------------------------------------------------------
# Tool 11 — score_variant_at
# ----------------------------------------------------------------------


@mcp.tool()
async def score_variant_at(
    chromosome: str,
    position: int,
    ref_base: str,
    alt_base: str,
    window_size: int = 8192,
    species: str = "human",
    assembly: str = "GRCh38",
    reduce_method: str = "mean",
) -> str:
    """Score a SNP by coordinate: fetch reference context then compute LL delta.

    USE THIS WHEN:
    - You have a variant as (chromosome, position, ref, alt) — the most common
      form from VCF, VEP output, or clinical reports
    - You want the pathogenicity signal in one tool call rather than a
      manual fetch + score chain
    - User asks "is this variant pathogenic?" with a coordinate

    DO NOT USE WHEN:
    - You only have a sequence in hand (no coordinate) → use `score_snp`
    - You need to score many variants → use `score_variant_batch` (after
      fetching contexts) or loop this tool — note Ensembl rate limit applies
    - Variant is at a splice boundary → use `score_splice_region` after
      fetching context

    PARAMETERS:
    - `chromosome`: chromosome name (e.g. "16", "X"). `chr` prefix is stripped.
    - `position`: 1-based genomic coordinate of the variant.
    - `ref_base`: single nucleotide (A/C/G/T/N) — must match the reference at
      `position` on the chosen `assembly` (validated; mismatch → error).
    - `alt_base`: single nucleotide — the alternative allele to score.
    - `window_size`: flanking window in bp (default 8192, Arc Institute method).
    - `species`: Ensembl species slug (default "human").
    - `assembly`: "GRCh38" (default) or "GRCh37".
    - `reduce_method`: "mean" (default, robust) or "sum".

    OUTPUT:
    - `score_delta = LL(alt) − LL(ref)`, both averaged per-position over the
      window with `reduce_method='mean'`.
    - For an 8 kb window on the 40B model, observed magnitudes are small:
      typically 1e-4 to 1e-3 absolute. The sign indicates direction; the
      magnitude is NOT a calibrated pathogenicity score. No published
      threshold maps Evo2 deltas to ACMG categories.
    - Zero-shot AUROC on Arc Institute's BRCA1 LOF benchmark is 0.73 for
      Evo2-1B (Brixi et al 2025); the 40B is improved but still imperfect.
      Treat the score as one piece of evidence, not a verdict.

    Returns markdown with LL(ref), LL(alt), delta, the genomic context used,
    and an explicit reminder that this is not a clinical classifier.
    """
    t0 = time.monotonic()

    if len(ref_base) != 1 or len(alt_base) != 1:
        return _err("ref_base and alt_base must each be exactly one nucleotide.")

    cache_key = cache.score_variant_at_key(
        chromosome, position, ref_base, alt_base, window_size, species, assembly, reduce_method
    )
    cached = cache.get("score_variant_at", cache_key)
    if cached is not None:
        return formatters.dump({
            "variant": cached["variant"],
            "context": cached["context"],
            "scores": cached["scores"],
            "strand_swap_note": cached.get("strand_swap_note"),
            "cache": {"hit": True, "saved_server_ms": cached.get("server_ms")},
            "runtime": formatters.runtime(0, (time.monotonic() - t0) * 1000.0),
        })
    if ref_base.upper() not in {"A", "C", "G", "T", "N"}:
        return _err(f"Invalid ref_base {ref_base!r}. Use A/C/G/T/N.")
    if alt_base.upper() not in {"A", "C", "G", "T", "N"}:
        return _err(f"Invalid alt_base {alt_base!r}. Use A/C/G/T/N.")

    ref_base_u = ref_base.upper()
    alt_base_u = alt_base.upper()

    try:
        ctx: FetchedContext = await _get_ensembl().fetch_variant_context(
            chromosome,
            position,
            window_size=window_size,
            species=species,
            assembly=assembly,
        )
    except (EnsemblError, ValueError) as exc:
        return _err(exc, stage="ensembl_fetch_context")

    observed = ctx.sequence[ctx.center_index]
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    strand_swap_note = None
    if observed != ref_base_u:
        # Common cause: variant was reported on the gene's coding strand but
        # the gene is on the reverse strand, so Ensembl returns the complement
        # on the forward strand. Auto-recover if (comp(ref), comp(alt)) matches.
        if complement.get(ref_base_u) == observed:
            ref_was = (ref_base_u, alt_base_u)
            ref_base_u = complement[ref_base_u]
            alt_base_u = complement[alt_base_u]
            strand_swap_note = (
                f"Note: Ensembl reference at `{ctx.chromosome}:{position}` "
                f"({ctx.assembly}) is `{observed}`, not `{ref_was[0]}`. The pair "
                f"`{ref_was[0]}>{ref_was[1]}` is the **reverse-complement** "
                f"convention (common for variants reported on a gene's coding "
                f"strand when the gene is on the negative strand of the genome). "
                f"Scored the forward-strand equivalent `{ref_base_u}>{alt_base_u}` instead — "
                f"this gives the SAME biological variant under Evo2 (which is "
                f"strand-agnostic for likelihood scoring). If you specifically "
                f"need the user-provided strand, double-check the gene's strand "
                f"in VEP / Ensembl before interpreting."
            )
        else:
            return _err(
                "Reference allele mismatch: neither the supplied ref nor its complement "
                "matches the Ensembl base at that position.",
                stage="reference_validation",
                supplied_ref=ref_base_u,
                observed_base=observed,
                supplied_ref_complement=complement.get(ref_base_u),
                chromosome=ctx.chromosome,
                position=position,
                assembly=ctx.assembly,
                window={"start": ctx.start, "end": ctx.end, "length": len(ctx.sequence)},
                likely_causes=[
                    "wrong assembly (position itself differs between GRCh37 and GRCh38)",
                    "off-by-one / 0-based vs 1-based coordinate convention",
                    "variant lifted-over from a different reference",
                ],
                suggestions=[
                    "try the other assembly",
                    "look up the rsID in dbSNP for canonical forward-strand coordinates",
                ],
            )

    try:
        mutated, mut_pos = apply_snp(ctx.sequence, alt_base_u, position=ctx.center_index)
        nim = _get_client()
        ref_logits, r_ms = await _forward_logits(nim, ctx.sequence)
        alt_logits, a_ms = await _forward_logits(nim, mutated)
        score_ref = log_likelihood_from_logits(
            ref_logits, ctx.sequence, reduce_method=reduce_method
        )
        score_alt = log_likelihood_from_logits(
            alt_logits, mutated, reduce_method=reduce_method
        )
    except (NimError, NimNotReadyError, NpzDecodeError, ScoringError) as exc:
        return _err(exc, stage="nim_scoring")

    result = {
        "variant": {
            "chromosome": ctx.chromosome,
            "position": ctx.start + ctx.center_index,
            "ref": ref_base_u,
            "alt": alt_base_u,
            "assembly": ctx.assembly,
        },
        "context": {
            "chromosome": ctx.chromosome,
            "start": ctx.start,
            "end": ctx.end,
            "length": len(ctx.sequence),
        },
        "scores": {
            "score_ref": score_ref,
            "score_alt": score_alt,
            "score_delta": score_alt - score_ref,
            "reduce_method": reduce_method,
        },
        "strand_swap_note": strand_swap_note,
    }
    cache.put("score_variant_at", cache_key, {**result, "server_ms": r_ms + a_ms})
    return formatters.dump({
        **result,
        "cache": {"hit": False},
        "runtime": formatters.runtime(r_ms + a_ms, (time.monotonic() - t0) * 1000.0),
    })
