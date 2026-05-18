"""Catalog of `output_layers` names accepted by the NIM `/forward` endpoint.

The NIM container does not expose an introspection endpoint listing valid
layer names, but the OpenAPI schema documents them. Names below are
confirmed against `nvcr.io/nim/arc/evo2:2` (NIM 2.1.0, model 2.0.0).

Conventions
-----------
- Request name vs response key: NIM always appends `.output` to the
  requested layer name in the NPZ archive it returns. e.g. request
  `output_layer` → NPZ key `output_layer.output`. Use `response_key()`.
- LM_HEAD: the final logits layer (shape `(seq_len, 1, 512)`).
  → used by `score_sequence`, `score_snp`, `score_variant_batch`,
    `score_splice_region`. Vocab size 512.
- DEFAULT_EMBEDDING_LAYER: a hidden state useful as a generic embedding.
  NIM doc recommends the final MLP of an intermediate layer (capture
  context-dependent features). Middle of the stack for the 40B model.
- RECOMMENDED_EMBEDDING_LAYERS: a curated subset surfaced to LLMs via
  `list_layer_names`.

Architecture (per NIM /forward OpenAPI):
- 7B: 32 layers (TransformerLayers at 3,10,17,24,31; HyenaLayers elsewhere)
- 40B: 50 layers (TransformerLayers at 3,10,17,24,31,35,42,49; Hyena elsewhere)
- Hidden dim: 8192 for 40B
"""

from __future__ import annotations

# Layer name producing the LM head logits (final unembedding).
# Confirmed against NIM 2.1.0 on Arkane H200 (2026-05-18).
LM_HEAD_LAYER = "output_layer"

# Default embedding layer: final MLP of a mid-stack layer, per NIM docs.
# Hyena layer (not 3/10/17/24/31/35/42/49 — those are Transformer).
DEFAULT_EMBEDDING_LAYER_40B = "decoder.layers.20.mlp"
DEFAULT_EMBEDDING_LAYER_7B = "decoder.layers.16.mlp"


def response_key(request_name: str) -> str:
    """NIM appends `.output` to every requested layer name in the NPZ response.

    Use this to look up the array after `decode_forward_response`.
    """
    return f"{request_name}.output"


# Curated list surfaced via `list_layer_names`
RECOMMENDED_LAYERS_40B: list[dict[str, str]] = [
    {
        "name": "decoder.layers.20.mlp",
        "purpose": "general-purpose embeddings, classification (mid-stack Hyena MLP)",
        "shape_hint": "(seq_len, 1, 8192)",
    },
    {
        "name": "decoder.layers.30.mlp",
        "purpose": "high-level features, language-modelling-aware embeddings",
        "shape_hint": "(seq_len, 1, 8192)",
    },
    {
        "name": "decoder.final_norm",
        "purpose": "final normalised representations (closest to LM head)",
        "shape_hint": "(seq_len, 1, 8192)",
    },
    {
        "name": "output_layer",
        "purpose": "next-token logits (used internally for scoring)",
        "shape_hint": "(seq_len, 1, 512)",
    },
]

RECOMMENDED_LAYERS_7B: list[dict[str, str]] = [
    {
        "name": "decoder.layers.16.mlp",
        "purpose": "general-purpose embeddings (mid-stack Hyena MLP)",
        "shape_hint": "(seq_len, 1, hidden_dim)",
    },
    {
        "name": "decoder.layers.24.mlp",
        "purpose": "higher-level features",
        "shape_hint": "(seq_len, 1, hidden_dim)",
    },
    {
        "name": "decoder.final_norm",
        "purpose": "final normalised representations",
        "shape_hint": "(seq_len, 1, hidden_dim)",
    },
    {
        "name": "output_layer",
        "purpose": "next-token logits (used internally for scoring)",
        "shape_hint": "(seq_len, 1, 512)",
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
