"""Integration tests for the FastMCP server tools.

These tests mock the NimClient and verify each tool wires through the full
pipeline (HTTP → NPZ decode → scoring → formatter) and produces the expected
markdown shape. They DO NOT require a running NIM container.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from evo2_nim_mcp import layer_catalog, server
from tests.conftest import npz_to_b64


@pytest.fixture
def mock_client():
    """A mock NimClient that returns whatever payloads are queued on its forward/generate methods."""
    client = AsyncMock()
    client.base_url = "http://test.nim"
    return client


@pytest.fixture(autouse=True)
def reset_global_client():
    """Each test starts with no cached client."""
    server._client = None
    yield
    server._client = None


def _logits_favoring(sequence: str, vocab_size: int = 512) -> np.ndarray:
    """Build (seq_len, vocab) logits where each position favors the actual next token."""
    n = len(sequence)
    logits = np.full((n, vocab_size), -10.0, dtype=np.float32)
    encoded = sequence.encode("ascii")
    for i in range(n - 1):
        logits[i, encoded[i + 1]] = 10.0
    return logits


def _forward_response(layer_name: str, array: np.ndarray, elapsed_ms: int = 100) -> dict[str, Any]:
    return {
        "data": npz_to_b64({layer_name: array}),
        "elapsed_ms": elapsed_ms,
    }


class TestScoreSequence:
    @pytest.mark.asyncio
    async def test_returns_markdown_with_score(self, mock_client):
        seq = "ACGTACGT"
        mock_client.forward.return_value = _forward_response(
            layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)
        )
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_sequence(seq)
        assert "# Sequence likelihood score" in out
        assert "8 nt" in out
        # Score should be high (favoring logits → near-zero log-likelihood)
        # We can't assert the exact value but it should be > -1
        # Extract the score line
        score_line = next(line for line in out.splitlines() if line.startswith("- **score**"))
        score_value = float(score_line.split(":")[1].split("(")[0].strip())
        assert score_value > -1.0

    @pytest.mark.asyncio
    async def test_handles_nim_error(self, mock_client):
        from evo2_nim_mcp.client import NimError
        mock_client.forward.side_effect = NimError("connection refused")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_sequence("ACGT")
        assert "# Error" in out
        assert "connection refused" in out

    @pytest.mark.asyncio
    async def test_handles_missing_lm_head_layer(self, mock_client):
        # NIM returns an NPZ with a different layer name
        mock_client.forward.return_value = _forward_response(
            "wrong_layer_name", np.zeros((4, 512), dtype=np.float32)
        )
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_sequence("ACGT")
        assert "# Error" in out
        assert "lm_head" in out.lower() or "wrong_layer" in out


class TestScoreSnp:
    @pytest.mark.asyncio
    async def test_returns_markdown_with_delta(self, mock_client):
        seq = "ACGTACGTAC"
        # Same logits for ref and alt → delta should be near zero
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
            _forward_response(
                layer_catalog.LM_HEAD_LAYER,
                _logits_favoring(seq[:5] + "G" + seq[6:]),
            ),
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp(seq, "G")
        assert "# SNP score" in out
        assert "score_delta" in out
        assert "position 5" in out  # center of 10-nt sequence

    @pytest.mark.asyncio
    async def test_invalid_allele_returns_error(self, mock_client):
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp("ACGTACGT", "X")
        assert "# Error" in out
        assert "A/C/G/T/N" in out

    @pytest.mark.asyncio
    async def test_short_sequence_returns_error(self, mock_client):
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp("AC", "G")
        assert "# Error" in out
        assert ">= 3" in out

    @pytest.mark.asyncio
    async def test_explicit_position(self, mock_client):
        seq = "ACGTACGTAC"
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
            _forward_response(
                layer_catalog.LM_HEAD_LAYER,
                _logits_favoring(seq[:2] + "T" + seq[3:]),
            ),
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp(seq, "T", position=2)
        assert "position 2" in out
        assert "G → T" in out


class TestScoreVariantBatch:
    @pytest.mark.asyncio
    async def test_processes_all_variants(self, mock_client):
        # 3 variants × 2 forward calls each = 6 forward responses queued
        seq = "ACGTACGTAC"
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq))
            for _ in range(6)
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_variant_batch(
                [
                    {"sequence": seq, "alternative_allele": "G", "id": "v1"},
                    {"sequence": seq, "alternative_allele": "T", "id": "v2"},
                    {"sequence": seq, "alternative_allele": "C", "id": "v3"},
                ]
            )
        assert "variants scored**: 3/3" in out
        assert "failures**: 0" in out

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_abort(self, mock_client):
        seq = "ACGTACGTAC"
        # Variant 1 succeeds, variant 2 has bad input (will fail in apply_snp)
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_variant_batch(
                [
                    {"sequence": seq, "alternative_allele": "G", "id": "v1"},
                    {"sequence": "AC", "alternative_allele": "G", "id": "v2"},  # too short
                ]
            )
        assert "variants scored**: 1/2" in out
        assert "failures**: 1" in out


class TestScoreSpliceRegion:
    @pytest.mark.asyncio
    async def test_canonical_donor_disruption(self, mock_client):
        seq = "A" * 20 + "GT" + "A" * 20
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
            _forward_response(
                layer_catalog.LM_HEAD_LAYER,
                _logits_favoring(seq[:20] + "GC" + seq[22:]),
            ),
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_splice_region(seq, 20, "GT", "GC")
        assert "splice" in out.lower()
        assert "GT → GC" in out
        assert "canonical" in out.lower()

    @pytest.mark.asyncio
    async def test_mismatched_reference_returns_error(self, mock_client):
        # The sequence at position 20 is "GT" but we claim it's "AC"
        seq = "A" * 20 + "GT" + "A" * 20
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_splice_region(seq, 20, "AC", "GG")
        assert "# Error" in out
        assert "does not match" in out

    @pytest.mark.asyncio
    async def test_invalid_dinucleotide_length_returns_error(self, mock_client):
        seq = "A" * 20 + "GT" + "A" * 20
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_splice_region(seq, 20, "G", "GG")  # ref is 1 char
        assert "# Error" in out
        assert "exactly 2" in out


class TestEmbedSequence:
    @pytest.mark.asyncio
    async def test_returns_markdown_with_shape(self, mock_client):
        seq = "ACGTACGT"
        embedding = np.random.default_rng(0).standard_normal((8, 4096)).astype(np.float32)
        mock_client.forward.return_value = _forward_response("blocks.20.output", embedding)
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.embed_sequence(seq)
        assert "Sequence embedding" in out
        assert "(8, 4096)" in out
        assert "blocks.20.output" in out

    @pytest.mark.asyncio
    async def test_explicit_layer_name(self, mock_client):
        seq = "ACGT"
        embedding = np.zeros((4, 1024), dtype=np.float32)
        mock_client.forward.return_value = _forward_response("custom.layer", embedding)
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.embed_sequence(seq, layer_name="custom.layer")
        assert "custom.layer" in out

    @pytest.mark.asyncio
    async def test_missing_layer_in_response_returns_error(self, mock_client):
        mock_client.forward.return_value = _forward_response(
            "wrong.layer", np.zeros((4, 4096), dtype=np.float32)
        )
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.embed_sequence("ACGT", layer_name="blocks.20.output")
        assert "# Error" in out
        assert "list_layer_names" in out


class TestGenerateSequence:
    @pytest.mark.asyncio
    async def test_returns_markdown_with_generation(self, mock_client):
        mock_client.generate.return_value = {
            "sequence": "ACGTACGTACGTACGT",
            "elapsed_ms": 2000,
        }
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.generate_sequence("ACGT", n_tokens=12)
        assert "# Sequence generation" in out
        assert "16 nt" in out
        assert "ACGTACGTACGTACGT" in out

    @pytest.mark.asyncio
    async def test_passes_random_seed_when_provided(self, mock_client):
        mock_client.generate.return_value = {"sequence": "AC", "elapsed_ms": 100}
        with patch.object(server, "_get_client", return_value=mock_client):
            await server.generate_sequence("ACGT", random_seed=42)
        call_payload = mock_client.generate.call_args[0][0]
        assert call_payload["random_seed"] == 42

    @pytest.mark.asyncio
    async def test_no_random_seed_when_not_provided(self, mock_client):
        mock_client.generate.return_value = {"sequence": "AC", "elapsed_ms": 100}
        with patch.object(server, "_get_client", return_value=mock_client):
            await server.generate_sequence("ACGT")
        call_payload = mock_client.generate.call_args[0][0]
        assert "random_seed" not in call_payload


class TestListAvailableCheckpoints:
    @pytest.mark.asyncio
    async def test_default_is_40b(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            out = await server.list_available_checkpoints()
        assert "evo2_40b" in out
        assert "40B" in out

    @pytest.mark.asyncio
    async def test_explicit_7b_variant(self) -> None:
        with patch.dict("os.environ", {"NIM_VARIANT": "7b"}, clear=True):
            out = await server.list_available_checkpoints()
        assert "evo2_7b" in out
        assert "7B" in out


class TestListLayerNames:
    @pytest.mark.asyncio
    async def test_includes_lm_head(self) -> None:
        with patch.dict("os.environ", {"NIM_VARIANT": "40b"}, clear=True):
            out = await server.list_layer_names()
        assert "lm_head.output" in out

    @pytest.mark.asyncio
    async def test_40b_recommended_layers_visible(self) -> None:
        with patch.dict("os.environ", {"NIM_VARIANT": "40b"}, clear=True):
            out = await server.list_layer_names()
        assert "blocks.20.output" in out


class TestNimHealth:
    @pytest.mark.asyncio
    async def test_ready(self, mock_client):
        mock_client.health.return_value = {"status": "ready", "version": "2.1.0"}
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.nim_health()
        assert "ready" in out
        assert "version" in out
        assert "2.1.0" in out

    @pytest.mark.asyncio
    async def test_not_ready(self, mock_client):
        from evo2_nim_mcp.client import NimNotReadyError
        mock_client.health.side_effect = NimNotReadyError("model still loading")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.nim_health()
        assert "not ready" in out

    @pytest.mark.asyncio
    async def test_unreachable(self, mock_client):
        from evo2_nim_mcp.client import NimError
        mock_client.health.side_effect = NimError("connection refused")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.nim_health()
        assert "unreachable" in out
