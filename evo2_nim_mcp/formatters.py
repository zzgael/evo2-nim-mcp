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
    if score_delta < -0.5:
        interp = "mutated sequence is markedly less likely under the model (consistent with deleterious effect)"
    elif score_delta < -0.1:
        interp = "mutated sequence is mildly less likely (weak deleterious signal)"
    elif score_delta <= 0.1:
        interp = "mutation is approximately neutral under the model"
    else:
        interp = "mutated sequence is more likely than the reference (rare; double-check input)"

    short_ref = sequence if len(sequence) <= 60 else f"{sequence[:30]}…{sequence[-20:]}"
    short_alt = (
        mutated_sequence
        if len(mutated_sequence) <= 60
        else f"{mutated_sequence[:30]}…{mutated_sequence[-20:]}"
    )
    return f"""# SNP score — position {position}, {reference_allele} → {alternative_allele}

## Result
- **score_delta**: {score_delta:+.4f} ({reduce_method} log-likelihood)
- **interpretation**: {interp}
- **reference score**: {score_ref:.4f}
- **mutated score**: {score_alt:.4f}

## Sequences
- ref: `{short_ref}`
- alt: `{short_alt}`

{_runtime_section(server_ms, total_ms)}
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

    rows = ["| # | position | ref | alt | score_delta | interpretation |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(results):
        if "error" in r:
            rows.append(f"| {i} | — | — | — | — | error: {r['error']} |")
            continue
        delta = r["score_delta"]
        if delta < -0.5:
            note = "deleterious"
        elif delta < -0.1:
            note = "mild deleterious"
        elif delta <= 0.1:
            note = "neutral"
        else:
            note = "gain (rare)"
        rows.append(
            f"| {i} | {r['position']} | {r['reference_allele']} | "
            f"{r['alternative_allele']} | {delta:+.4f} | {note} |"
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
    canonical_note = "canonical donor (GT) or acceptor (AG)" if canonical else "non-canonical splice motif"
    if score_delta < -0.5:
        interp = "splice site disruption is severely scored (consistent with splice loss)"
    elif score_delta < -0.1:
        interp = "splice site disruption shows mild deleterious signal"
    else:
        interp = "splice site change is roughly neutral under Evo2"

    return f"""# Splice region score — position {splice_position}, {reference_dinucleotide} → {alternative_dinucleotide}

## Result
- **score_delta**: {score_delta:+.4f}
- **interpretation**: {interp}
- **motif type**: {canonical_note}
- **reference score**: {score_ref:.4f}
- **mutated score**: {score_alt:.4f}

## Region
- splice position: {splice_position}
- region length: {len(sequence)} nt

{_runtime_section(server_ms, total_ms)}
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
