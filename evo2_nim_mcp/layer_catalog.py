"""Catalog of `output_layers` names accepted by the NIM `/forward` endpoint.

The NIM container does not expose an introspection endpoint listing valid
layer names. This catalog is curated empirically — names are confirmed by
sending a tiny `/forward` request and checking the response payload.

When deploying against a new NIM version, run the discovery script in
`docs/nim-layer-names.md` and update this file.

Conventions
-----------
- LM_HEAD: the layer that produces vocab logits (shape (seq_len, 512))
  → used by `score_sequence`, `score_snp`, `score_variant_batch`,
    `score_splice_region`.
- DEFAULT_EMBEDDING_LAYER: a hidden state useful as a generic embedding,
  selected for general-purpose downstream classification per Evo2 paper
  guidance ("intermediate layers like Block 20 (40B) often perform best").
- RECOMMENDED_EMBEDDING_LAYERS: a curated subset to surface to LLMs via
  `list_layer_names` so they know which layers are useful for which tasks.
"""

from __future__ import annotations

# These are placeholders until the Arkane trial confirms the actual layer names.
# The values below are educated guesses based on Evo2 architecture conventions
# (Hyena layers organized in blocks, byte-level vocab of 512). The trial will
# either confirm them or replace them.

# Layer name producing the LM head logits
LM_HEAD_LAYER = "lm_head.output"

# Default embedding layer for general-purpose use (40B model: middle block)
DEFAULT_EMBEDDING_LAYER_40B = "blocks.20.output"
DEFAULT_EMBEDDING_LAYER_7B = "blocks.16.output"

# Curated list surfaced via `list_layer_names`
RECOMMENDED_LAYERS_40B: list[dict[str, str]] = [
    {
        "name": "blocks.20.output",
        "purpose": "general-purpose embeddings, classification",
        "shape_hint": "(seq_len, 4096)",
    },
    {
        "name": "blocks.30.output",
        "purpose": "high-level features, language-modelling-aware embeddings",
        "shape_hint": "(seq_len, 4096)",
    },
    {
        "name": "lm_head.output",
        "purpose": "next-token logits (used internally for scoring)",
        "shape_hint": "(seq_len, 512)",
    },
]

RECOMMENDED_LAYERS_7B: list[dict[str, str]] = [
    {
        "name": "blocks.2.mlp.l3",
        "purpose": "low-level features (upstream evo2-mcp default)",
        "shape_hint": "(seq_len, hidden_dim)",
    },
    {
        "name": "blocks.16.output",
        "purpose": "general-purpose embeddings",
        "shape_hint": "(seq_len, hidden_dim)",
    },
    {
        "name": "lm_head.output",
        "purpose": "next-token logits (used internally for scoring)",
        "shape_hint": "(seq_len, 512)",
    },
]


def recommended_for(checkpoint: str) -> list[dict[str, str]]:
    """Return the recommended layer catalog for a given checkpoint name."""
    if "40b" in checkpoint.lower():
        return RECOMMENDED_LAYERS_40B
    if "7b" in checkpoint.lower():
        return RECOMMENDED_LAYERS_7B
    # Conservative default: surface the most generic 40B catalog
    return RECOMMENDED_LAYERS_40B


def default_embedding_layer(checkpoint: str) -> str:
    """The layer name to use when the caller does not specify one."""
    if "40b" in checkpoint.lower():
        return DEFAULT_EMBEDDING_LAYER_40B
    if "7b" in checkpoint.lower():
        return DEFAULT_EMBEDDING_LAYER_7B
    return DEFAULT_EMBEDDING_LAYER_40B
