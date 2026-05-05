"""Unit tests for evo2_nim_mcp.scoring."""

from __future__ import annotations

import numpy as np
import pytest

from evo2_nim_mcp.scoring import (
    ScoringError,
    apply_snp,
    encode_sequence_bytes,
    log_likelihood_from_logits,
    log_softmax,
)


class TestEncodeSequenceBytes:
    def test_uppercase_dna_maps_to_ascii(self) -> None:
        result = encode_sequence_bytes("ACGT")
        np.testing.assert_array_equal(result, np.array([65, 67, 71, 84], dtype=np.int64))

    def test_lowercase_dna_also_works(self) -> None:
        result = encode_sequence_bytes("acgt")
        np.testing.assert_array_equal(result, np.array([97, 99, 103, 116], dtype=np.int64))

    def test_n_supported(self) -> None:
        result = encode_sequence_bytes("AN")
        np.testing.assert_array_equal(result, np.array([65, 78], dtype=np.int64))

    def test_empty_raises(self) -> None:
        with pytest.raises(ScoringError, match="empty sequence"):
            encode_sequence_bytes("")

    def test_non_ascii_raises_with_position(self) -> None:
        with pytest.raises(ScoringError) as excinfo:
            encode_sequence_bytes("ACGT€T")
        msg = str(excinfo.value)
        assert "position 4" in msg
        assert "non-ASCII" in msg

    def test_returns_int64(self) -> None:
        assert encode_sequence_bytes("A").dtype == np.int64


class TestLogSoftmax:
    def test_simple_case(self) -> None:
        x = np.array([[1.0, 2.0, 3.0]])
        result = log_softmax(x)
        # log_softmax([1, 2, 3]) = log(exp([1,2,3]) / sum(exp([1,2,3])))
        # = [1, 2, 3] - logsumexp([1, 2, 3])
        expected_logsumexp = np.log(np.exp(1) + np.exp(2) + np.exp(3))
        expected = np.array([[1 - expected_logsumexp, 2 - expected_logsumexp, 3 - expected_logsumexp]])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_sum_of_exp_is_one(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, 100))
        result = log_softmax(x, axis=-1)
        # exp(log_softmax).sum(axis=-1) should be 1.0
        np.testing.assert_allclose(np.exp(result).sum(axis=-1), 1.0, rtol=1e-10)

    def test_numerical_stability_with_large_values(self) -> None:
        # Without the max-subtraction trick this would overflow
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = log_softmax(x)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(np.exp(result).sum(), 1.0, rtol=1e-10)

    def test_negative_inf_handling(self) -> None:
        x = np.array([[-np.inf, 0.0, 0.0]])
        result = log_softmax(x)
        # First entry should be -inf; others should be log(0.5) ≈ -0.693
        assert result[0, 0] == -np.inf
        np.testing.assert_allclose(result[0, 1:], np.log(0.5), rtol=1e-10)

    def test_returns_float64(self) -> None:
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        result = log_softmax(x)
        assert result.dtype == np.float64


class TestLogLikelihoodFromLogits:
    def _build_logits_favoring(self, sequence: str, vocab_size: int = 512) -> np.ndarray:
        """Build logits where each position strongly favors the next token of `sequence`."""
        n = len(sequence)
        token_ids = encode_sequence_bytes(sequence)
        logits = np.full((n, vocab_size), -10.0, dtype=np.float32)
        # logits[i] should predict token i+1; for i in 0..n-2 set the favored token
        for i in range(n - 1):
            logits[i, token_ids[i + 1]] = 10.0
        return logits

    def test_perfect_prediction_gives_high_log_likelihood(self) -> None:
        seq = "ACGTACGT"
        logits = self._build_logits_favoring(seq)
        ll = log_likelihood_from_logits(logits, seq, reduce_method="mean")
        # Each position's prediction is essentially deterministic → log prob ≈ 0
        assert ll > -0.01

    def test_uniform_logits_yields_log_one_over_vocab(self) -> None:
        seq = "ACGT"
        logits = np.zeros((4, 512), dtype=np.float32)  # uniform over 512
        ll = log_likelihood_from_logits(logits, seq, reduce_method="mean")
        np.testing.assert_allclose(ll, np.log(1 / 512), rtol=1e-6)

    def test_sum_vs_mean_reduce_method(self) -> None:
        seq = "ACGTACGT"  # 8 tokens → 7 positions scored
        logits = self._build_logits_favoring(seq)
        ll_sum = log_likelihood_from_logits(logits, seq, reduce_method="sum")
        ll_mean = log_likelihood_from_logits(logits, seq, reduce_method="mean")
        # mean = sum / 7
        np.testing.assert_allclose(ll_mean, ll_sum / 7, rtol=1e-6)

    def test_invalid_reduce_method_raises(self) -> None:
        with pytest.raises(ScoringError, match="reduce_method must be"):
            log_likelihood_from_logits(
                np.zeros((4, 512)), "ACGT", reduce_method="median"  # type: ignore
            )

    def test_mismatched_seq_dim_raises(self) -> None:
        logits = np.zeros((10, 512))
        with pytest.raises(ScoringError, match="seq dim"):
            log_likelihood_from_logits(logits, "ACGT")

    def test_too_short_sequence_raises(self) -> None:
        with pytest.raises(ScoringError, match="at least 2 tokens"):
            log_likelihood_from_logits(np.zeros((1, 512)), "A")

    def test_logits_must_be_2d(self) -> None:
        logits = np.zeros((4, 512, 1))
        with pytest.raises(ScoringError, match="2D"):
            log_likelihood_from_logits(logits, "ACGT")

    def test_vocab_too_small_for_byte_alphabet(self) -> None:
        # Vocab size 50 cannot encode 'T' (token id 84)
        logits = np.zeros((4, 50))
        with pytest.raises(ScoringError, match="vocab size"):
            log_likelihood_from_logits(logits, "ACGT")

    def test_alt_lower_than_ref_when_logits_disagree(self) -> None:
        ref = "ACGTACGT"
        alt = "ACGTAGGT"  # changed position 5: C → G
        logits = self._build_logits_favoring(ref)
        ll_ref = log_likelihood_from_logits(logits, ref, reduce_method="mean")
        ll_alt = log_likelihood_from_logits(logits, alt, reduce_method="mean")
        assert ll_alt < ll_ref  # mutated sequence should score worse under ref-favoring logits


class TestApplySnp:
    def test_center_position_default(self) -> None:
        mutated, pos = apply_snp("ACGTACGT", "G")  # len 8, center = 4
        assert pos == 4
        assert mutated == "ACGTGCGT"

    def test_explicit_position(self) -> None:
        mutated, pos = apply_snp("ACGTACGT", "T", position=2)
        assert pos == 2
        assert mutated == "ACTTACGT"

    def test_too_short_raises(self) -> None:
        with pytest.raises(ScoringError, match=">= 3"):
            apply_snp("AC", "G")

    def test_multi_char_allele_raises(self) -> None:
        with pytest.raises(ScoringError, match="exactly 1 nucleotide"):
            apply_snp("ACGTACGT", "GA")

    def test_invalid_allele_char_raises(self) -> None:
        with pytest.raises(ScoringError, match="must be one of A/C/G/T/N"):
            apply_snp("ACGTACGT", "X")

    def test_position_out_of_range_raises(self) -> None:
        with pytest.raises(ScoringError, match="out of range"):
            apply_snp("ACGTACGT", "G", position=99)

    def test_negative_position_raises(self) -> None:
        with pytest.raises(ScoringError, match="out of range"):
            apply_snp("ACGTACGT", "G", position=-1)

    def test_identity_mutation_is_allowed(self) -> None:
        # Replacing C with C at the center position — valid, just a no-op
        mutated, pos = apply_snp("ACGTACGT", "A", position=4)
        assert mutated == "ACGTACGT"
        assert pos == 4

    def test_lowercase_allele_normalized(self) -> None:
        # 'a' is allowed (upper-case validation), and the mutated sequence preserves it
        mutated, _ = apply_snp("ACGTACGT", "a", position=4)
        assert mutated == "ACGTaCGT"
