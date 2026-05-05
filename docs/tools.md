# Tool reference

Nine tools exposed by the MCP server. Every tool returns a markdown report.

## `score_sequence(sequence, reduce_method="mean")`

Whole-sequence log-likelihood under Evo2.

**Use when** the user wants to rank sequences by biological plausibility or screen designed sequences for naturalness.

**Don't use when** the user has a specific point mutation (`score_snp`), multiple variants (`score_variant_batch`), or a splice variant (`score_splice_region`).

**Returns** a single scalar score. Higher (less negative) = more plausible. With `reduce_method="mean"`, scores near `log(1/4) ≈ -1.39` correspond to roughly random nucleotides; natural genomic DNA typically scores higher.

## `score_snp(sequence, alternative_allele, reduce_method="mean", position=None)`

Score a single nucleotide variant: log-likelihood of reference vs mutated sequence, returning the delta.

**Use when** the user has a specific point mutation and wants a quantitative effect score.

**Don't use when** the user has multiple variants (`score_variant_batch`) or a splice variant (`score_splice_region`).

**Returns** ref score, mutated score, and `score_delta = score_alt - score_ref`. Delta < 0 means mutated less likely (often deleterious). The mutation is applied at `position` (default: center of the sequence).

## `score_variant_batch(variants, reduce_method="mean")`

Batch-score multiple SNPs in one call.

**Use when** the user has a cohort or VCF-like list of point variants and wants a ranked table.

**Don't use when** there's just one variant (`score_snp`) or splice-aware scoring is needed (`score_splice_region` per variant).

**Input shape**: `list[{sequence, alternative_allele, position?, id?}]`

**Returns** a markdown table with one row per variant (sorted in input order). Per-row failures are reported without aborting the batch.

## `score_splice_region(sequence, splice_position, reference_dinucleotide, alternative_dinucleotide, reduce_method="mean")`

Score a splice site change at a dinucleotide boundary.

**Use when** a variant affects a canonical splice donor (`GT`) or acceptor (`AG`) at the boundary of an intron.

**Don't use when** the variant is in a coding region away from splice boundaries (`score_snp`).

**Returns** ref/mut scores, delta, and a flag indicating whether the reference dinucleotide is a canonical splice motif.

## `embed_sequence(sequence, layer_name=None)`

Extract embeddings at a named hidden layer.

**Use when** the user wants vector representations for similarity search, clustering, or downstream classification.

**Don't use when** the user wants a scalar score (`score_sequence`).

**Returns** a markdown summary with embedding shape and L2 norm statistics. The full tensor is not returned in markdown — use the NIM `/forward` endpoint directly if you need raw values.

If `layer_name` is None, the default for the loaded checkpoint is used (intermediate transformer block; see `list_layer_names`).

## `generate_sequence(prompt, n_tokens=100, temperature=0.7, top_k=4, top_p=0.0, random_seed=None)`

Generate a DNA continuation conditioned on a prompt.

**Use when** the user wants Evo2 to design or extend a DNA sequence (workflow D — DNA element design).

**Don't use when** the user wants to score an existing sequence (`score_sequence`) or extract embeddings (`embed_sequence`).

**Returns** the generated continuation. Note: this is a plausible continuation under Evo2, NOT a verified biological design — downstream validation required.

## `list_available_checkpoints()`

List the model checkpoint loaded in this NIM deployment.

**Use when** the LLM is unsure which Evo2 model is loaded (40B vs 7B), or for sanity-checking unexpected results.

A NIM container exposes exactly one checkpoint at a time. To switch, redeploy with a different `NIM_VARIANT`.

## `list_layer_names()`

List recommended `output_layers` names accepted by `/forward` for the loaded checkpoint.

**Use when** the LLM is about to call `embed_sequence` and needs to know which layer to use, or when a previous call returned an "unknown layer" error.

The catalog is curated empirically — names are confirmed by sending small `/forward` requests. Missing names may still work; this is a recommended list, not exhaustive.

## `nim_health()`

Check whether the NIM container is reachable and the model is loaded.

**Use when** other tool calls fail with connection / timeout errors, or as a sanity check before kicking off a batch.

**Returns** status (`ready` / `not ready` / `unreachable`), the NIM URL, and any extra metadata the NIM exposes.
