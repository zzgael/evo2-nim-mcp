"""Compute log-likelihoods and SNP score deltas from Evo2 NIM `/forward` outputs.

The NIM returns raw layer tensors; this module turns them into per-token log
probabilities and aggregated scores. All math is done in float64 to keep
edge-case precision; the input tensors are typically float32 or float16.

Vocabulary
----------
Evo2 uses a byte-level character tokenizer. For DNA, each nucleotide is its
ASCII byte value:
    'A' → 65   'a' → 97
    'C' → 67   'c' → 99
    'G' → 71   'g' → 103
    'T' → 84   't' → 116
    'N' → 78   'n' → 110

The model's vocab size is 512 (extended from 256 raw bytes by special tokens).
We rely on the byte-level mapping for the standard nucleotide alphabet only;
the trial will confirm this matches NIM behaviour. If the trial reveals a
different mapping, override `BYTE_VOCAB` here.
"""

from __future__ import annotations

import numpy as np

VOCAB_SIZE = 512  # per NVIDIA NIM docs


class ScoringError(Exception):
    """Raised when the math cannot be performed (wrong shapes, unknown chars, etc.)."""


def encode_sequence_bytes(sequence: str) -> np.ndarray:
    """Map each character to its byte (ASCII) token ID. Returns int64 array of shape (len,).

    Raises ScoringError if any character has a code point > 255 (out of byte range).
    """
    if not sequence:
        raise ScoringError("Cannot encode an empty sequence.")
    try:
        # encode to bytes then to ints — single-byte UTF-8 encoded chars only
        encoded = sequence.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ScoringError(
            f"Sequence contains non-ASCII characters at position {exc.start}: "
            f"{sequence[exc.start : exc.end]!r}. "
            "Evo2 expects DNA in IUPAC ASCII (A,C,G,T,N)."
        ) from exc
    return np.frombuffer(encoded, dtype=np.uint8).astype(np.int64)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax. Operates in float64 for precision."""
    x = logits.astype(np.float64, copy=False)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def log_likelihood_from_logits(
    logits: np.ndarray,
    sequence: str,
    *,
    reduce_method: str = "mean",
) -> float:
    """Compute log-likelihood of `sequence` under autoregressive logits.

    Parameters
    ----------
    logits : np.ndarray
        Shape (seq_len, vocab_size). The LM head outputs of the NIM `/forward` endpoint.
        Convention: logits[i] is the prediction over the token at position i+1
        (next-token prediction). We score positions 1..seq_len-1, scoring (seq_len - 1)
        positions in total.
    sequence : str
        The DNA sequence whose tokens were fed to the model.
    reduce_method : str
        "mean" (default) averages per-position log probs; "sum" totals them.

    Returns
    -------
    float
        Log-likelihood. Higher = more plausible under the model.

    Raises
    ------
    ScoringError
        If shapes don't line up or the sequence is too short.
    """
    if reduce_method not in {"mean", "sum"}:
        raise ScoringError(
            f"reduce_method must be 'mean' or 'sum', got {reduce_method!r}."
        )

    token_ids = encode_sequence_bytes(sequence)
    seq_len = token_ids.shape[0]

    if logits.ndim != 2:
        raise ScoringError(
            f"logits must be 2D (seq_len, vocab_size); got shape {logits.shape}."
        )
    if logits.shape[0] != seq_len:
        raise ScoringError(
            f"logits seq dim ({logits.shape[0]}) does not match sequence length ({seq_len})."
        )
    if logits.shape[1] < int(token_ids.max()) + 1:
        raise ScoringError(
            f"logits vocab size ({logits.shape[1]}) is smaller than required for the byte alphabet "
            f"(max token id seen: {int(token_ids.max())}). The model output may not be the LM head."
        )

    if seq_len < 2:
        raise ScoringError(
            "Need at least 2 tokens to compute autoregressive log-likelihood."
        )

    # Score tokens 1..n-1 against logits 0..n-2 (next-token prediction).
    log_probs_all = log_softmax(logits, axis=-1)  # (seq_len, vocab)
    target_log_probs = log_probs_all[np.arange(seq_len - 1), token_ids[1:]]  # (seq_len - 1,)

    if reduce_method == "sum":
        return float(target_log_probs.sum())
    return float(target_log_probs.mean())


def per_position_log_likelihoods(
    logits: np.ndarray,
    sequence: str,
) -> np.ndarray:
    """Return the per-position log-likelihood vector under autoregressive scoring.

    Output shape: `(seq_len - 1,)`, dtype `float32`. Element i is the
    log-probability of token i+1 under the model's distribution at position i
    (next-token prediction). Aggregating with `.mean()` or `.sum()` gives
    the same scalar as `log_likelihood_from_logits(...)` with the matching
    `reduce_method`.

    Useful for per-base analysis: locating bases the model finds unlikely
    under the reference, plotting likelihood profiles, ref-vs-alt
    position-wise comparison.

    Raises the same `ScoringError` cases as `log_likelihood_from_logits`.
    """
    token_ids = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(np.int64)
    seq_len = token_ids.shape[0]

    if logits.ndim != 2:
        raise ScoringError(
            f"logits must be 2D (seq_len, vocab_size); got shape {logits.shape}."
        )
    if logits.shape[0] != seq_len:
        raise ScoringError(
            f"logits seq dim ({logits.shape[0]}) does not match sequence length ({seq_len})."
        )
    if seq_len < 2:
        raise ScoringError(
            "Need at least 2 tokens to compute autoregressive log-likelihood."
        )

    log_probs_all = log_softmax(logits, axis=-1)
    return log_probs_all[np.arange(seq_len - 1), token_ids[1:]].astype(np.float32)


def apply_snp(sequence: str, alternative_allele: str, position: int | None = None) -> tuple[str, int]:
    """Return (mutated_sequence, position) where the base at `position` is replaced by `alternative_allele`.

    If `position` is None, the center index is used (consistent with the upstream
    `score_snp` convention from the original evo2-mcp).
    """
    if len(sequence) < 3:
        raise ScoringError("score_snp requires a reference sequence of length >= 3.")
    if len(alternative_allele) != 1:
        raise ScoringError(
            f"alternative_allele must be exactly 1 nucleotide, got {alternative_allele!r}."
        )
    if alternative_allele.upper() not in {"A", "C", "G", "T", "N"}:
        raise ScoringError(
            f"alternative_allele must be one of A/C/G/T/N, got {alternative_allele!r}."
        )

    if position is None:
        position = len(sequence) // 2
    if not 0 <= position < len(sequence):
        raise ScoringError(
            f"position {position} is out of range for sequence of length {len(sequence)}."
        )

    if sequence[position].upper() == alternative_allele.upper():
        # Mutation is identity — flag for caller, but still return a valid output
        pass

    mutated = sequence[:position] + alternative_allele + sequence[position + 1 :]
    return mutated, position
