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

from evo2_nim_mcp import formatters, layer_catalog
from evo2_nim_mcp.client import NimClient, NimError, NimNotReadyError
from evo2_nim_mcp.ensembl import EnsemblClient, EnsemblError, FetchedContext
from evo2_nim_mcp.npz import NpzDecodeError, decode_forward_response
from evo2_nim_mcp.scoring import (
    ScoringError,
    apply_snp,
    log_likelihood_from_logits,
)

mcp = FastMCP("Evo2 NIM")

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
async def score_sequence(sequence: str, reduce_method: str = "mean") -> str:
    """Compute the log-likelihood of a DNA sequence under Evo2.

    USE THIS WHEN:
    - User wants to rank candidate sequences by biological plausibility
    - User wants a single scalar score for a whole sequence
    - User is screening designed/synthetic sequences for naturalness

    DO NOT USE WHEN:
    - User has a specific point mutation → prefer `score_snp` (gives a delta)
    - User has multiple variants to score → prefer `score_variant_batch`
    - User has a splice variant → prefer `score_splice_region`

    PARAMETERS:
    - `sequence`: DNA, length ≥ 2, IUPAC alphabet (A, C, G, T, N — upper or lowercase).
    - `reduce_method`: "mean" (default; robust to length) or "sum" (totals all positions).

    INTERPRETATION:
    - Higher (less negative) score = more plausible under the model.
    - The score is a per-position log-likelihood (with `reduce_method="mean"`).
      A score near `log(1/4) ≈ -1.39` corresponds to roughly random nucleotides;
      natural genomic DNA typically scores higher (less negative).

    Returns a markdown report with the score, sequence preview, and runtime.
    """
    t0 = time.monotonic()
    try:
        logits, server_ms = await _forward_logits(_get_client(), sequence)
        score = log_likelihood_from_logits(logits, sequence, reduce_method=reduce_method)
    except (NimError, NimNotReadyError, NpzDecodeError, ScoringError) as exc:
        return f"# Error\n\n{exc}"

    return formatters.format_score_sequence(
        sequence=sequence,
        score=score,
        reduce_method=reduce_method,
        server_ms=server_ms,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )


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

    INTERPRETATION:
    - `score_delta` < 0 → mutated sequence less likely (often deleterious).
    - `score_delta` ≈ 0 → mutation roughly neutral under the model.
    - `score_delta` > 0 → mutated sequence more likely (rare; double-check inputs).

    Returns markdown with reference score, mutated score, delta, and an
    interpretation hint.
    """
    t0 = time.monotonic()
    try:
        mutated, pos = apply_snp(sequence, alternative_allele, position=position)
        client = _get_client()
        ref_logits, ref_server_ms = await _forward_logits(client, sequence)
        alt_logits, alt_server_ms = await _forward_logits(client, mutated)
        score_ref = log_likelihood_from_logits(ref_logits, sequence, reduce_method=reduce_method)
        score_alt = log_likelihood_from_logits(alt_logits, mutated, reduce_method=reduce_method)
    except (NimError, NimNotReadyError, NpzDecodeError, ScoringError) as exc:
        return f"# Error\n\n{exc}"

    return formatters.format_score_snp(
        sequence=sequence,
        mutated_sequence=mutated,
        position=pos,
        reference_allele=sequence[pos],
        alternative_allele=alternative_allele,
        score_ref=score_ref,
        score_alt=score_alt,
        score_delta=score_alt - score_ref,
        reduce_method=reduce_method,
        server_ms=ref_server_ms + alt_server_ms,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )


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

    INTERPRETATION:
    - Output is a markdown table sorted in input order with one row per variant.
    - Each row carries `score_delta` and a short interpretation tag
      (deleterious / mild deleterious / neutral / gain).
    - Failures (bad inputs, NIM errors) are reported per-row without aborting
      the whole batch.

    Returns markdown with summary statistics and per-variant table.
    """
    t0 = time.monotonic()
    client = _get_client()
    results: list[dict[str, Any]] = []
    server_ms_total = 0.0
    for v in variants:
        try:
            seq = v["sequence"]
            alt = v["alternative_allele"]
            pos = v.get("position")
            mutated, position = apply_snp(seq, alt, position=pos)
            ref_logits, r_ms = await _forward_logits(client, seq)
            alt_logits, a_ms = await _forward_logits(client, mutated)
            score_ref = log_likelihood_from_logits(ref_logits, seq, reduce_method=reduce_method)
            score_alt = log_likelihood_from_logits(alt_logits, mutated, reduce_method=reduce_method)
            server_ms_total += r_ms + a_ms
            results.append(
                {
                    "id": v.get("id"),
                    "position": position,
                    "reference_allele": seq[position],
                    "alternative_allele": alt,
                    "score_ref": score_ref,
                    "score_alt": score_alt,
                    "score_delta": score_alt - score_ref,
                }
            )
        except (NimError, NimNotReadyError, NpzDecodeError, ScoringError, KeyError) as exc:
            results.append({"id": v.get("id"), "error": str(exc)})

    return formatters.format_score_variant_batch(
        results=results,
        reduce_method=reduce_method,
        server_ms=server_ms_total,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )


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

    INTERPRETATION:
    - The output flags whether the reference is a canonical splice motif (GT/AG).
    - A markedly negative `score_delta` on a canonical motif strongly suggests
      splice site disruption.

    Returns markdown with reference / mutated scores, delta, and motif interpretation.
    """
    t0 = time.monotonic()
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
        return f"# Error\n\n{exc}"

    return formatters.format_score_splice_region(
        sequence=sequence,
        splice_position=splice_position,
        reference_dinucleotide=reference_dinucleotide,
        alternative_dinucleotide=alternative_dinucleotide,
        score_ref=score_ref,
        score_alt=score_alt,
        score_delta=score_alt - score_ref,
        canonical=canonical,
        server_ms=r_ms + a_ms,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )


# ----------------------------------------------------------------------
# Tool 5 — embed_sequence
# ----------------------------------------------------------------------


@mcp.tool()
async def embed_sequence(sequence: str, layer_name: str | None = None) -> str:
    """Extract embeddings for a DNA sequence at a named hidden layer.

    USE THIS WHEN:
    - User wants vector representations for similarity search, clustering,
      or downstream classification
    - User is building a feature matrix from many sequences

    DO NOT USE WHEN:
    - User wants a single scalar score → use `score_sequence`
    - User does not specify a layer and you don't know which one is appropriate
      → call `list_layer_names` first

    PARAMETERS:
    - `sequence`: DNA, IUPAC alphabet, length ≥ 1
    - `layer_name`: name of an `output_layers` entry to extract.
      If None, a sensible default for the loaded checkpoint is used
      (intermediate block; see `list_layer_names`).

    INTERPRETATION:
    - The output reports the embedding's shape and the L2 norm distribution
      (mean ± std across positions). Wildly different norms across sequences
      may indicate unusual or out-of-distribution input.
    - The full embedding tensor is NOT returned in markdown for size reasons —
      only summary statistics. Callers needing the raw tensor should call the
      NIM `/forward` endpoint directly.

    Returns markdown summary.
    """
    t0 = time.monotonic()
    checkpoint = _checkpoint_name()
    layer = layer_name or layer_catalog.default_embedding_layer(checkpoint)
    try:
        client = _get_client()
        response = await client.forward({"sequence": sequence, "output_layers": [layer]})
        arrays = decode_forward_response(response.get("data", ""))
        key = layer_catalog.response_key(layer)
        if key not in arrays:
            return (
                f"# Error\n\nNIM /forward did not return key {key!r}. "
                f"Available keys in response: {list(arrays)}. "
                f"Call `list_layer_names` to see the catalog of supported layer names."
            )
        embedding = arrays[key]
        # NIM returns (seq_len, batch=1, hidden); squeeze batch to (seq_len, hidden).
        if embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(axis=1)
        norms = np.linalg.norm(embedding.astype(np.float64), axis=-1)
        summary_rows = ", ".join(
            f"{float(embedding[i, 0]):+.3f}" for i in range(min(5, embedding.shape[0]))
        )
    except (NimError, NimNotReadyError, NpzDecodeError) as exc:
        return f"# Error\n\n{exc}"

    return formatters.format_embed_sequence(
        sequence=sequence,
        layer_name=layer,
        embedding_shape=tuple(embedding.shape),
        norm_mean=float(norms.mean()),
        norm_std=float(norms.std()),
        server_ms=float(response.get("elapsed_ms", 0.0)),
        total_ms=(time.monotonic() - t0) * 1000.0,
        embedding_summary=f"first column, first 5 positions: [{summary_rows}]",
    )


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
        return f"# Error\n\n{exc}"

    generated = response.get("sequence", "")
    server_ms = float(response.get("elapsed_ms", 0.0))
    return formatters.format_generate_sequence(
        prompt=prompt,
        generated=generated,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        server_ms=server_ms,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )


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
    return formatters.format_list_checkpoints(
        [{"name": checkpoint, "description": description}]
    )


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
    return formatters.format_list_layer_names(checkpoint, layers)


# ----------------------------------------------------------------------
# Tool 9 — nim_health
# ----------------------------------------------------------------------


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
        return formatters.format_nim_health(
            status="ready",
            base_url=client.base_url,
            extra={k: v for k, v in body.items() if k != "status"} or None,
        )
    except NimNotReadyError as exc:
        return formatters.format_nim_health(
            status=f"not ready ({exc})", base_url=client.base_url
        )
    except NimError as exc:
        return formatters.format_nim_health(
            status=f"unreachable ({exc})", base_url=client.base_url
        )


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
        return f"# Error\n\n{exc}"

    return formatters.format_fetch_variant_context(
        ctx=ctx,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )


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

    INTERPRETATION:
    - `score_delta = log_likelihood(alt_window) - log_likelihood(ref_window)`,
      averaged over the window with `reduce_method='mean'`.
    - For ~8 kb windows on the 40B model, observed magnitudes are small:
      ~1e-4 to 1e-3 absolute. More negative → more disruptive.
    - Heuristic ranges (combine with VEP, AlphaGenome, structure, literature):
        `delta < -3e-4`: strong disruption signal
        `-3e-4 ≤ delta < -1e-4`: moderate signal
        `-1e-4 ≤ delta ≤ +1e-4`: weak/uncertain (VUS)
        `delta > +1e-4`: variant scores higher than reference (rare; double-check)
    - Do NOT use Evo2 alone for clinical classification.

    Returns markdown with reference/mutated scores, delta, interpretation
    band, and the genomic context used.
    """
    t0 = time.monotonic()

    if len(ref_base) != 1 or len(alt_base) != 1:
        return "# Error\n\n`ref_base` and `alt_base` must each be exactly one nucleotide."
    if ref_base.upper() not in {"A", "C", "G", "T", "N"}:
        return f"# Error\n\nInvalid `ref_base` {ref_base!r}. Use A/C/G/T/N."
    if alt_base.upper() not in {"A", "C", "G", "T", "N"}:
        return f"# Error\n\nInvalid `alt_base` {alt_base!r}. Use A/C/G/T/N."

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
        return f"# Error fetching genomic context\n\n{exc}"

    observed = ctx.sequence[ctx.center_index]
    if observed != ref_base_u:
        return (
            f"# Reference allele mismatch\n\n"
            f"You provided `ref_base={ref_base_u!r}` at `{ctx.chromosome}:{position}` "
            f"({ctx.assembly}), but Ensembl returned `{observed!r}` at that position.\n\n"
            "Likely causes:\n"
            "- Wrong assembly (GRCh37 vs GRCh38)\n"
            "- Off-by-one or 0-based vs 1-based coordinate convention\n"
            "- Strand convention (Evo2 scores the forward strand of the reference)\n\n"
            f"Window fetched: `{ctx.chromosome}:{ctx.start}..{ctx.end}` ({len(ctx.sequence)} bp)."
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
        return f"# Error scoring variant\n\n{exc}"

    return formatters.format_score_variant_at(
        ctx=ctx,
        ref_base=ref_base_u,
        alt_base=alt_base_u,
        score_ref=score_ref,
        score_alt=score_alt,
        score_delta=score_alt - score_ref,
        reduce_method=reduce_method,
        server_ms=r_ms + a_ms,
        total_ms=(time.monotonic() - t0) * 1000.0,
    )
