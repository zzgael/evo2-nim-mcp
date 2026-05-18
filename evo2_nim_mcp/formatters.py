"""LLM-friendly markdown formatters for tool responses.

Each formatter takes the structured result of a tool call and produces a
markdown string suitable for an LLM to summarise back to a user. The format
is consistent across tools:

    # <Tool name> — <one-line subject>

    ## Result
    - structured key/value bullets, with interpretation hint where applicable

    ## <Section relevant to the tool>

    ## Runtime
    - server-side: <ms>
    - total: <ms>
"""

from __future__ import annotations

from typing import Any


def _runtime_section(server_ms: float | None, total_ms: float | None) -> str:
    lines = ["## Runtime"]
    if server_ms is not None:
        lines.append(f"- server-side: {server_ms:.0f} ms")
    if total_ms is not None:
        lines.append(f"- total: {total_ms:.0f} ms")
    return "\n".join(lines)


def format_score_sequence(
    *,
    sequence: str,
    score: float,
    reduce_method: str,
    server_ms: float | None,
    total_ms: float | None,
) -> str:
    short = sequence if len(sequence) <= 60 else f"{sequence[:30]}…{sequence[-20:]}"
    return f"""# Sequence likelihood score

## Result
- **score**: {score:.4f} ({reduce_method} log-likelihood per position)
- **sequence length**: {len(sequence)} nt
- **interpretation**: higher (less negative) = more plausible under Evo2

## Sequence
- preview: `{short}`

{_runtime_section(server_ms, total_ms)}
"""


def format_score_snp(
    *,
    sequence: str,
    mutated_sequence: str,
    position: int,
    reference_allele: str,
    alternative_allele: str,
    score_ref: float,
    score_alt: float,
    score_delta: float,
    reduce_method: str,
    server_ms: float | None,
    total_ms: float | None,
) -> str:
    short_ref = sequence if len(sequence) <= 60 else f"{sequence[:30]}…{sequence[-20:]}"
    short_alt = (
        mutated_sequence
        if len(mutated_sequence) <= 60
        else f"{mutated_sequence[:30]}…{mutated_sequence[-20:]}"
    )
    return f"""# SNP score — position {position}, {reference_allele} → {alternative_allele}

## Result
- **score_delta**: {score_delta:+.6f}
- **LL(ref)**: {score_ref:.6f}
- **LL(alt)**: {score_alt:.6f}
- **reduce_method**: `{reduce_method}` (per-position log-likelihood)
- **window length**: {len(sequence)} nt

## Sequences
- ref: `{short_ref}`
- alt: `{short_alt}`

{_runtime_section(server_ms, total_ms)}

Sign convention: `delta = LL(alt) − LL(ref)`. The model is not a clinical
classifier; interpret the score in conjunction with VEP consequence,
AlphaGenome, structural prediction, population frequency, and literature.
"""


def format_score_variant_batch(
    *,
    results: list[dict[str, Any]],
    reduce_method: str,
    server_ms: float | None,
    total_ms: float | None,
) -> str:
    n = len(results)
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    rows = [
        "| # | position | ref | alt | LL(ref) | LL(alt) | score_delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results):
        if "error" in r:
            rows.append(f"| {i} | — | — | — | — | — | error: {r['error']} |")
            continue
        rows.append(
            f"| {i} | {r['position']} | {r['reference_allele']} | "
            f"{r['alternative_allele']} | {r['score_ref']:.6f} | "
            f"{r['score_alt']:.6f} | {r['score_delta']:+.6f} |"
        )

    summary = (
        f"- **variants scored**: {len(successes)}/{n}\n"
        f"- **failures**: {len(failures)}\n"
        f"- **reduce_method**: {reduce_method}"
    )

    return f"""# SNP batch scoring — {n} variants

## Summary
{summary}

## Per-variant results
{chr(10).join(rows)}

{_runtime_section(server_ms, total_ms)}
"""


def format_score_splice_region(
    *,
    sequence: str,
    splice_position: int,
    reference_dinucleotide: str,
    alternative_dinucleotide: str,
    score_ref: float,
    score_alt: float,
    score_delta: float,
    canonical: bool,
    server_ms: float | None,
    total_ms: float | None,
) -> str:
    canonical_note = (
        "canonical donor (GT) or acceptor (AG)"
        if canonical
        else "non-canonical motif (not a standard GT/AG splice site)"
    )

    return f"""# Splice region score — position {splice_position}, {reference_dinucleotide} → {alternative_dinucleotide}

## Result
- **score_delta**: {score_delta:+.6f}
- **LL(ref)**: {score_ref:.6f}
- **LL(alt)**: {score_alt:.6f}
- **motif type**: {canonical_note}

## Region
- splice position: {splice_position}
- region length: {len(sequence)} nt

{_runtime_section(server_ms, total_ms)}

Sign convention: `delta = LL(alt) − LL(ref)`. The model is not a splice-site
classifier — for splice-effect prediction, cross-reference with SpliceAI,
AlphaGenome's splice track, and the variant's distance to the canonical
donor/acceptor.
"""


def format_embed_sequence(
    *,
    sequence: str,
    layer_name: str,
    embedding_shape: tuple[int, ...],
    norm_mean: float,
    norm_std: float,
    server_ms: float | None,
    total_ms: float | None,
    embedding_summary: str | None = None,
) -> str:
    sample_section = f"## Sample values\n{embedding_summary}\n\n" if embedding_summary else ""
    return f"""# Sequence embedding — layer `{layer_name}`

## Result
- **shape**: {embedding_shape}
- **L2 norm — mean**: {norm_mean:.3f}
- **L2 norm — std**: {norm_std:.3f}
- **layer**: `{layer_name}`

## Sequence
- length: {len(sequence)} nt

{sample_section}{_runtime_section(server_ms, total_ms)}
"""


def format_generate_sequence(
    *,
    prompt: str,
    generated: str,
    n_tokens: int,
    temperature: float,
    top_k: int,
    server_ms: float | None,
    total_ms: float | None,
) -> str:
    short_prompt = prompt if len(prompt) <= 60 else f"{prompt[:30]}…{prompt[-20:]}"
    short_gen = generated if len(generated) <= 200 else f"{generated[:120]}…{generated[-60:]}"
    return f"""# Sequence generation

## Result
- **generated length**: {len(generated)} nt
- **temperature**: {temperature}
- **top_k**: {top_k}

## Prompt
`{short_prompt}`

## Generated
```
{short_gen}
```

{_runtime_section(server_ms, total_ms)}
"""


def format_list_checkpoints(checkpoints: list[dict[str, Any]]) -> str:
    rows = ["| name | description |", "|---|---|"]
    for cp in checkpoints:
        rows.append(f"| `{cp.get('name', '?')}` | {cp.get('description', '—')} |")
    return f"""# Available checkpoints

{chr(10).join(rows)}

A NIM container exposes exactly one checkpoint at a time, configured at deploy time
via the `NIM_VARIANT` environment variable. To switch checkpoints, redeploy the container.
"""


def format_list_layer_names(checkpoint: str, layers: list[dict[str, str]]) -> str:
    rows = ["| name | purpose | shape hint |", "|---|---|---|"]
    for layer in layers:
        rows.append(
            f"| `{layer.get('name', '?')}` | {layer.get('purpose', '—')} | "
            f"{layer.get('shape_hint', '—')} |"
        )
    return f"""# Available `output_layers` for checkpoint `{checkpoint}`

{chr(10).join(rows)}

Use these names verbatim when calling `embed_sequence(layer_name=...)` or
`score_sequence` (which internally requests `lm_head.output`).
"""


def format_fetch_variant_context(
    *,
    ctx,
    total_ms: float | None,
) -> str:
    """Markdown summary of a successful Ensembl region fetch.

    `ctx` is `evo2_nim_mcp.ensembl.FetchedContext`. Typed loosely to avoid an
    import cycle in the formatters module.
    """
    seq = ctx.sequence
    n = len(seq)
    preview_radius = 30
    if n <= 2 * preview_radius:
        preview = seq
    else:
        c = ctx.center_index
        left = max(0, c - preview_radius)
        right = min(n, c + preview_radius + 1)
        preview = f"…{seq[left:c]}[{seq[c]}]{seq[c+1:right]}…"

    # FASTA-style wrap so the sequence is readable AND copy/paste-able into
    # downstream tool calls (embed_similarity, score_snp). Don't omit anything.
    LINE = 80
    wrapped = "\n".join(seq[i : i + LINE] for i in range(0, n, LINE))

    return f"""# Fetched genomic context

## Result
- **species**: {ctx.species}
- **assembly**: {ctx.assembly}
- **region**: `{ctx.chromosome}:{ctx.start}..{ctx.end}` ({n} bp returned)
- **center_index**: {ctx.center_index} (0-based offset inside the returned sequence)
- **reference base at center**: `{seq[ctx.center_index]}`

## Sequence preview (centered on variant position, bracketed = reference base)
```
{preview}
```

## Full reference sequence ({n} bp)
The complete sequence is below in FASTA-wrapped form. To use it in another
tool (e.g. `embed_similarity`, `score_snp`), concatenate the lines into a
single string with no newlines.

```
{wrapped}
```

{_runtime_section(server_ms=None, total_ms=total_ms)}

Downstream usage:
- `score_snp(sequence=<full above>, alternative_allele='X', position={ctx.center_index})` — score a specific SNV at the centered coordinate.
- `score_variant_at(chromosome='{ctx.chromosome}', position={ctx.start + ctx.center_index}, ref_base='{seq[ctx.center_index]}', alt_base='X', assembly='{ctx.assembly}')` — one-call equivalent (re-fetches internally).
- `embed_similarity(sequence_a=<this window>, sequence_b=<another window>)` — pairwise embedding comparison.
"""


def format_score_variant_at(
    *,
    ctx,
    ref_base: str,
    alt_base: str,
    score_ref: float,
    score_alt: float,
    score_delta: float,
    reduce_method: str,
    server_ms: float | None,
    total_ms: float | None,
    strand_swap_note: str | None = None,
) -> str:
    """Markdown summary of `score_variant_at` (fetch + score combined)."""
    swap_section = ""
    if strand_swap_note:
        swap_section = f"\n## ⚠ Strand convention auto-corrected\n{strand_swap_note}\n"
    return f"""# Evo2 variant score

## Variant
- **{ctx.chromosome}:{ctx.start + ctx.center_index} {ref_base}>{alt_base}** ({ctx.assembly}, forward strand)
- **context window**: `{ctx.chromosome}:{ctx.start}..{ctx.end}` ({len(ctx.sequence)} bp)
- **reduce_method**: `{reduce_method}` (per-position log-likelihood, averaged over the window)
{swap_section}

## Scores
- **LL(ref)**: {score_ref:.6f}
- **LL(alt)**: {score_alt:.6f}
- **delta** (`LL(alt) − LL(ref)`): {score_delta:+.6f}

{_runtime_section(server_ms, total_ms)}

Evo2 returns a likelihood score — not a clinical classifier. The published
zero-shot AUROC on the BRCA1 LOF vs FUNC/INT benchmark is 0.73 for Evo2-1B
(Brixi et al 2025); 40B is improved but not perfect. Interpretation requires
the clinician's judgement combined with VEP consequence, AlphaGenome
regulatory/splice signal, AlphaFold structural impact, population frequency,
and the existing literature on the variant.
"""


def format_embed_similarity(
    *,
    sequence_a_len: int,
    sequence_b_len: int,
    layer_name: str,
    cosine_pool: float,
    cosine_position_mean: float | None,
    server_ms: float | None,
    total_ms: float | None,
) -> str:
    """Markdown summary of `embed_similarity`."""
    pos_line = (
        f"- **cosine_similarity_centered** (position-wise mean): {cosine_position_mean:+.4f}"
        if cosine_position_mean is not None
        else "- **cosine_similarity_centered**: not reported (sequences differ in length)"
    )
    return f"""# Embedding similarity

## Result
- **layer**: `{layer_name}`
- **sequence A length**: {sequence_a_len} nt
- **sequence B length**: {sequence_b_len} nt
- **cosine_similarity_mean_pool**: {cosine_pool:+.4f}  (mean-pool each embedding, then cosine of the two pooled vectors)
{pos_line}

## What cosine similarity means
- Range: -1 (anti-parallel) to +1 (identical). 0 = orthogonal.
- `cosine_similarity_mean_pool`: mean of the embedding tensor across
  positions, then cosine of the two pooled vectors. Compares the *overall*
  representation of each sequence.
- `cosine_similarity_centered` (when lengths match): cosine at each position
  averaged across positions. More sensitive to local differences.

No published threshold maps cosine values to functional / clinical
categories. Compare across a panel of variants (e.g. each variant vs the
wild-type window) to rank by divergence; the absolute magnitude alone is
not actionable.

{_runtime_section(server_ms, total_ms)}
"""


def format_nim_health(
    *,
    status: str,
    base_url: str,
    extra: dict[str, Any] | None = None,
) -> str:
    extra_lines = ""
    if extra:
        extra_lines = "\n".join(f"- **{k}**: {v}" for k, v in extra.items())
    return f"""# NIM health

## Result
- **status**: {status}
- **endpoint**: `{base_url}`
{extra_lines}
"""
