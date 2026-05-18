"""Unit tests for evo2_nim_mcp.layer_catalog."""

from __future__ import annotations

from evo2_nim_mcp import layer_catalog


class TestRecommendedFor:
    def test_40b_returns_40b_catalog(self) -> None:
        layers = layer_catalog.recommended_for("evo2_40b")
        assert layers is layer_catalog.RECOMMENDED_LAYERS_40B
        assert any(
            layer["name"] == layer_catalog.DEFAULT_EMBEDDING_LAYER_40B
            for layer in layers
        )

    def test_7b_returns_7b_catalog(self) -> None:
        layers = layer_catalog.recommended_for("evo2_7b")
        assert layers is layer_catalog.RECOMMENDED_LAYERS_7B
        assert any(
            layer["name"] == layer_catalog.DEFAULT_EMBEDDING_LAYER_7B
            for layer in layers
        )

    def test_unknown_falls_back_to_40b(self) -> None:
        layers = layer_catalog.recommended_for("evo2_unknown")
        assert layers is layer_catalog.RECOMMENDED_LAYERS_40B

    def test_case_insensitive(self) -> None:
        assert (
            layer_catalog.recommended_for("EVO2_40B")
            is layer_catalog.RECOMMENDED_LAYERS_40B
        )


class TestDefaultEmbeddingLayer:
    def test_40b_returns_decoder_mid_mlp(self) -> None:
        assert (
            layer_catalog.default_embedding_layer("evo2_40b")
            == "decoder.layers.20.mlp"
        )

    def test_7b_returns_decoder_mid_mlp(self) -> None:
        assert (
            layer_catalog.default_embedding_layer("evo2_7b")
            == "decoder.layers.16.mlp"
        )

    def test_unknown_falls_back_to_40b_default(self) -> None:
        assert (
            layer_catalog.default_embedding_layer("evo2_???")
            == layer_catalog.DEFAULT_EMBEDDING_LAYER_40B
        )


class TestCatalogShape:
    def test_each_entry_has_required_keys(self) -> None:
        for catalog in (
            layer_catalog.RECOMMENDED_LAYERS_40B,
            layer_catalog.RECOMMENDED_LAYERS_7B,
        ):
            for entry in catalog:
                assert "name" in entry
                assert "purpose" in entry
                assert "shape_hint" in entry

    def test_lm_head_present_in_both_catalogs(self) -> None:
        for catalog in (
            layer_catalog.RECOMMENDED_LAYERS_40B,
            layer_catalog.RECOMMENDED_LAYERS_7B,
        ):
            assert any(
                layer["name"] == layer_catalog.LM_HEAD_LAYER for layer in catalog
            )


class TestResponseKey:
    def test_appends_dot_output(self) -> None:
        # NIM always appends `.output` to the requested layer name in the NPZ.
        assert layer_catalog.response_key("output_layer") == "output_layer.output"
        assert (
            layer_catalog.response_key("decoder.layers.20.mlp")
            == "decoder.layers.20.mlp.output"
        )
