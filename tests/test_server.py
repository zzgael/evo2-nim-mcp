"""Integration tests for the FastMCP server tools.

These tests mock the NimClient and verify each tool wires through the full
pipeline (HTTP → NPZ decode → scoring → JSON serialise) and produces the
expected JSON shape. They DO NOT require a running NIM container.

Every tool returns a JSON string; the tests parse it back and inspect the
fields the LLM relies on.
"""

from __future__ import annotations

import json
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
    # The server looks up arrays by `layer_catalog.response_key(layer_name)`,
    # which appends `.output` to the canonical layer name. Match that so the
    # mocked NPZ exposes the same key the consumer expects.
    return {
        "data": npz_to_b64({layer_catalog.response_key(layer_name): array}),
        "elapsed_ms": elapsed_ms,
    }


def _parse(out: str) -> dict:
    """Every tool returns JSON now. Decode + give a useful failure if not."""
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool output is not JSON: {out[:300]!r}") from exc


class TestScoreSequence:
    @pytest.mark.asyncio
    async def test_returns_score_and_length(self, mock_client):
        seq = "ACGTACGT"
        mock_client.forward.return_value = _forward_response(
            layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)
        )
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_sequence(seq)
        d = _parse(out)
        assert d["sequence_length"] == 8
        assert d["reduce_method"] == "mean"
        assert d["score"] > -1.0  # favoring logits → high LL
        assert "runtime" in d

    @pytest.mark.asyncio
    async def test_handles_nim_error(self, mock_client):
        from evo2_nim_mcp.client import NimError
        mock_client.forward.side_effect = NimError("connection refused")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_sequence("ACGT")
        d = _parse(out)
        assert "error" in d
        assert "connection refused" in d["error"]["message"]

    @pytest.mark.asyncio
    async def test_handles_missing_lm_head_layer(self, mock_client):
        mock_client.forward.return_value = _forward_response(
            "wrong_layer_name", np.zeros((4, 512), dtype=np.float32)
        )
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_sequence("ACGT")
        d = _parse(out)
        assert "error" in d
        msg = d["error"]["message"].lower()
        assert "lm_head" in msg or "wrong_layer" in msg


class TestScoreSnp:
    @pytest.mark.asyncio
    async def test_returns_delta(self, mock_client):
        seq = "ACGTACGTAC"
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
            _forward_response(
                layer_catalog.LM_HEAD_LAYER,
                _logits_favoring(seq[:5] + "G" + seq[6:]),
            ),
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp(seq, "G")
        d = _parse(out)
        assert d["position"] == 5  # center of 10-nt
        assert d["alternative_allele"] == "G"
        assert "score_delta" in d
        assert "score_ref" in d and "score_alt" in d

    @pytest.mark.asyncio
    async def test_invalid_allele_returns_error(self, mock_client):
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp("ACGTACGT", "X")
        d = _parse(out)
        assert "error" in d
        assert "A/C/G/T/N" in d["error"]["message"]

    @pytest.mark.asyncio
    async def test_short_sequence_returns_error(self, mock_client):
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_snp("AC", "G")
        d = _parse(out)
        assert "error" in d
        assert ">= 3" in d["error"]["message"]

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
        d = _parse(out)
        assert d["position"] == 2
        assert d["reference_allele"] == "G"
        assert d["alternative_allele"] == "T"


class TestScoreVariantBatch:
    @pytest.mark.asyncio
    async def test_processes_all_variants(self, mock_client):
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
        d = _parse(out)
        assert d["n"] == 3
        assert d["n_success"] == 3
        assert d["n_failures"] == 0
        assert len(d["results"]) == 3

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_abort(self, mock_client):
        seq = "ACGTACGTAC"
        mock_client.forward.side_effect = [
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
            _forward_response(layer_catalog.LM_HEAD_LAYER, _logits_favoring(seq)),
        ]
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_variant_batch(
                [
                    {"sequence": seq, "alternative_allele": "G", "id": "v1"},
                    {"sequence": "AC", "alternative_allele": "G", "id": "v2"},
                ]
            )
        d = _parse(out)
        assert d["n"] == 2
        assert d["n_success"] == 1
        assert d["n_failures"] == 1


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
        d = _parse(out)
        assert d["splice_position"] == 20
        assert d["reference_dinucleotide"] == "GT"
        assert d["alternative_dinucleotide"] == "GC"
        assert d["canonical"] is True

    @pytest.mark.asyncio
    async def test_mismatched_reference_returns_error(self, mock_client):
        seq = "A" * 20 + "GT" + "A" * 20
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_splice_region(seq, 20, "AC", "GG")
        d = _parse(out)
        assert "error" in d
        assert "does not match" in d["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_dinucleotide_length_returns_error(self, mock_client):
        seq = "A" * 20 + "GT" + "A" * 20
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.score_splice_region(seq, 20, "G", "GG")
        d = _parse(out)
        assert "error" in d
        assert "exactly 2" in d["error"]["message"]


class TestEmbedSequence:
    @pytest.mark.asyncio
    async def test_returns_shape(self, mock_client):
        seq = "ACGTACGT"
        default_layer = layer_catalog.default_embedding_layer("evo2_40b")
        embedding = np.random.default_rng(0).standard_normal((8, 4096)).astype(np.float32)
        mock_client.forward.return_value = _forward_response(default_layer, embedding)
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.embed_sequence(seq)
        d = _parse(out)
        assert d["embedding_shape"] == [8, 4096]
        assert d["layer_name"] == default_layer
        assert d["sequence_length"] == 8

    @pytest.mark.asyncio
    async def test_explicit_layer_name(self, mock_client):
        seq = "ACGT"
        embedding = np.zeros((4, 1024), dtype=np.float32)
        mock_client.forward.return_value = _forward_response("custom.layer", embedding)
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.embed_sequence(seq, layer_name="custom.layer")
        d = _parse(out)
        assert d["layer_name"] == "custom.layer"

    @pytest.mark.asyncio
    async def test_missing_layer_in_response_returns_error(self, mock_client):
        # NPZ has key for `wrong.layer.output` but the server is asked for
        # `decoder.layers.20.mlp.output` — should surface a clean JSON error.
        mock_client.forward.return_value = _forward_response(
            "wrong.layer", np.zeros((4, 4096), dtype=np.float32)
        )
        default_layer = layer_catalog.default_embedding_layer("evo2_40b")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.embed_sequence("ACGT", layer_name=default_layer)
        d = _parse(out)
        assert "error" in d
        assert "list_layer_names" in d["error"].get("hint", "")


class TestGenerateSequence:
    @pytest.mark.asyncio
    async def test_returns_generation(self, mock_client):
        mock_client.generate.return_value = {
            "sequence": "ACGTACGTACGTACGT",
            "elapsed_ms": 2000,
        }
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.generate_sequence("ACGT", n_tokens=12)
        d = _parse(out)
        assert d["generated"] == "ACGTACGTACGTACGT"
        assert d["generated_length"] == 16
        assert d["n_tokens"] == 12

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
        d = _parse(out)
        assert d["checkpoints"][0]["name"] == "evo2_40b"
        assert "40B" in d["checkpoints"][0]["description"]

    @pytest.mark.asyncio
    async def test_explicit_7b_variant(self) -> None:
        with patch.dict("os.environ", {"NIM_VARIANT": "7b"}, clear=True):
            out = await server.list_available_checkpoints()
        d = _parse(out)
        assert d["checkpoints"][0]["name"] == "evo2_7b"
        assert "7B" in d["checkpoints"][0]["description"]


class TestListLayerNames:
    @pytest.mark.asyncio
    async def test_includes_lm_head(self) -> None:
        with patch.dict("os.environ", {"NIM_VARIANT": "40b"}, clear=True):
            out = await server.list_layer_names()
        d = _parse(out)
        names = [layer["name"] for layer in d["layers"]]
        # output_layer is the LM-head equivalent the catalog ships under that name
        assert any("output_layer" in n or "lm_head" in n for n in names)

    @pytest.mark.asyncio
    async def test_40b_recommended_layers_visible(self) -> None:
        with patch.dict("os.environ", {"NIM_VARIANT": "40b"}, clear=True):
            out = await server.list_layer_names()
        d = _parse(out)
        names = [layer["name"] for layer in d["layers"]]
        assert any("decoder.layers.20" in n or "blocks.20" in n for n in names)


class TestNimHealth:
    @pytest.mark.asyncio
    async def test_ready(self, mock_client):
        mock_client.health.return_value = {"status": "ready", "version": "2.1.0"}
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.nim_health()
        d = _parse(out)
        assert d["status"] == "ready"
        assert d["extra"] == {"version": "2.1.0"}

    @pytest.mark.asyncio
    async def test_not_ready(self, mock_client):
        from evo2_nim_mcp.client import NimNotReadyError
        mock_client.health.side_effect = NimNotReadyError("model still loading")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.nim_health()
        d = _parse(out)
        assert d["status"] == "not_ready"
        assert "model still loading" in d["extra"]["detail"]

    @pytest.mark.asyncio
    async def test_unreachable(self, mock_client):
        from evo2_nim_mcp.client import NimError
        mock_client.health.side_effect = NimError("connection refused")
        with patch.object(server, "_get_client", return_value=mock_client):
            out = await server.nim_health()
        d = _parse(out)
        assert d["status"] == "unreachable"
        assert "connection refused" in d["extra"]["detail"]
