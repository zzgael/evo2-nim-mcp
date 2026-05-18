"""Snapshot-style tests for evo2_nim_mcp.formatters.

These tests don't compare full strings — they assert that key structural
elements (headings, fields, interpretation hints) are present in the output,
so small wording tweaks don't fail the suite.
"""

from __future__ import annotations

from evo2_nim_mcp import formatters


class TestScoreSequence:
    def test_includes_score_and_length(self) -> None:
        out = formatters.format_score_sequence(
            sequence="ACGTACGT",
            score=-1.234,
            reduce_method="mean",
            server_ms=100,
            total_ms=110,
        )
        assert "# Sequence likelihood score" in out
        assert "-1.2340" in out
        assert "8 nt" in out
        assert "mean" in out
        assert "100 ms" in out
        assert "110 ms" in out

    def test_truncates_long_sequence_in_preview(self) -> None:
        long_seq = "A" * 100
        out = formatters.format_score_sequence(
            sequence=long_seq, score=0.0, reduce_method="mean", server_ms=1, total_ms=1
        )
        # Truncated form contains an ellipsis
        assert "…" in out
        # But the length field shows the real length
        assert "100 nt" in out


class TestScoreSnp:
    def _call(self, score_delta: float) -> str:
        score_alt = -1.0 + score_delta
        return formatters.format_score_snp(
            sequence="ACGTACGT",
            mutated_sequence="ACGTGCGT",
            position=4,
            reference_allele="A",
            alternative_allele="G",
            score_ref=-1.0,
            score_alt=score_alt,
            score_delta=score_delta,
            reduce_method="mean",
            server_ms=200,
            total_ms=210,
        )

    def test_reports_ll_ref_and_alt(self) -> None:
        out = self._call(score_delta=-1.0)
        assert "LL(ref)" in out
        assert "LL(alt)" in out
        assert "-1.000000" in out  # both LL values printed at .6f

    def test_no_clinical_interpretation_labels(self) -> None:
        """The formatter should NOT emit pseudo-clinical labels (deleterious,
        neutral, gain, VUS) since Evo2 is not a clinical classifier and no
        published threshold maps deltas to ACMG categories."""
        for delta in (-1.0, -0.3, 0.05, 0.5):
            out = self._call(score_delta=delta).lower()
            for label in ("deleterious", "mild", "neutral", "gain (rare)", "vus range"):
                assert label not in out, f"unexpected label '{label}' for delta={delta}"

    def test_explicit_not_a_classifier_disclaimer(self) -> None:
        out = self._call(score_delta=-1.0).lower()
        # disclaimer may wrap across lines — normalize whitespace
        normalized = " ".join(out.split())
        assert "not a clinical classifier" in normalized

    def test_includes_position_and_alleles_in_heading(self) -> None:
        out = self._call(score_delta=-1.0)
        assert "position 4" in out
        assert "A → G" in out

    def test_signed_delta_formatting(self) -> None:
        # +/- prefix for the delta (now .6f precision)
        assert "+0.500000" in self._call(0.5)
        assert "-0.500000" in self._call(-0.5)


class TestScoreVariantBatch:
    def test_summary_counts_failures(self) -> None:
        results = [
            {
                "id": "v1",
                "position": 4,
                "reference_allele": "A",
                "alternative_allele": "G",
                "score_ref": -1.0,
                "score_alt": -2.0,
                "score_delta": -1.0,
            },
            {"id": "v2", "error": "bad input"},
            {
                "id": "v3",
                "position": 4,
                "reference_allele": "C",
                "alternative_allele": "T",
                "score_ref": -1.0,
                "score_alt": -1.05,
                "score_delta": -0.05,
            },
        ]
        out = formatters.format_score_variant_batch(
            results=results, reduce_method="mean", server_ms=300, total_ms=310
        )
        assert "variants scored**: 2/3" in out
        assert "failures**: 1" in out
        # raw numbers only — no clinical labels
        for label in ("deleterious", "mild deleterious", "neutral", "gain (rare)"):
            assert label not in out.lower(), f"unexpected label '{label}'"
        # both LL columns and the delta should be present
        assert "LL(ref)" in out
        assert "LL(alt)" in out
        assert "bad input" in out

    def test_table_has_one_row_per_variant(self) -> None:
        results = [
            {
                "id": f"v{i}",
                "position": 4,
                "reference_allele": "A",
                "alternative_allele": "G",
                "score_ref": -1.0,
                "score_alt": -1.5,
                "score_delta": -0.5,
            }
            for i in range(5)
        ]
        out = formatters.format_score_variant_batch(
            results=results, reduce_method="mean", server_ms=1, total_ms=1
        )
        # Each row index 0..4 should appear in a table cell
        for i in range(5):
            assert f"| {i} |" in out


class TestScoreSpliceRegion:
    def test_canonical_donor_motif_called_out(self) -> None:
        out = formatters.format_score_splice_region(
            sequence="A" * 20 + "GT" + "A" * 20,
            splice_position=20,
            reference_dinucleotide="GT",
            alternative_dinucleotide="GC",
            score_ref=-1.0,
            score_alt=-2.5,
            score_delta=-1.5,
            canonical=True,
            server_ms=400,
            total_ms=420,
        )
        assert "canonical donor" in out.lower() or "canonical" in out.lower()
        # raw numbers, no clinical verdicts
        assert "LL(ref)" in out
        assert "LL(alt)" in out
        for label in ("splice loss", "deleterious", "mild"):
            assert label not in out.lower(), f"unexpected label '{label}'"

    def test_non_canonical_motif_called_out(self) -> None:
        out = formatters.format_score_splice_region(
            sequence="ACGTAC" + "TT" + "ACGT",
            splice_position=6,
            reference_dinucleotide="TT",
            alternative_dinucleotide="GG",
            score_ref=-1.0,
            score_alt=-1.05,
            score_delta=-0.05,
            canonical=False,
            server_ms=400,
            total_ms=420,
        )
        assert "non-canonical" in out.lower()


class TestEmbedSequence:
    def test_includes_layer_and_shape(self) -> None:
        out = formatters.format_embed_sequence(
            sequence="ACGTACGT",
            layer_name="blocks.20.output",
            embedding_shape=(8, 4096),
            norm_mean=12.5,
            norm_std=2.1,
            server_ms=500,
            total_ms=510,
            embedding_summary=None,
        )
        assert "blocks.20.output" in out
        assert "(8, 4096)" in out
        assert "12.500" in out
        assert "2.100" in out

    def test_includes_embedding_summary_when_provided(self) -> None:
        out = formatters.format_embed_sequence(
            sequence="ACGT",
            layer_name="blocks.20.output",
            embedding_shape=(4, 4096),
            norm_mean=10.0,
            norm_std=1.0,
            server_ms=100,
            total_ms=110,
            embedding_summary="first column: [+0.123, -0.456, +0.789]",
        )
        assert "first column" in out


class TestGenerateSequence:
    def test_includes_lengths_and_temperature(self) -> None:
        out = formatters.format_generate_sequence(
            prompt="ACGT",
            generated="ACGTACGTACGT",
            n_tokens=8,
            temperature=0.7,
            top_k=4,
            server_ms=2000,
            total_ms=2050,
        )
        assert "12 nt" in out
        assert "0.7" in out
        assert "top_k**: 4" in out
        assert "ACGTACGTACGT" in out

    def test_truncates_very_long_generated_sequence(self) -> None:
        out = formatters.format_generate_sequence(
            prompt="ACGT",
            generated="A" * 5000,
            n_tokens=5000,
            temperature=0.7,
            top_k=4,
            server_ms=10000,
            total_ms=10100,
        )
        assert "5000 nt" in out
        # Should not splat 5000 'A's into the markdown
        assert out.count("A" * 1000) == 0


class TestListCheckpoints:
    def test_renders_table(self) -> None:
        out = formatters.format_list_checkpoints(
            [{"name": "evo2_40b", "description": "Big model"}]
        )
        assert "| name | description |" in out
        assert "evo2_40b" in out
        assert "Big model" in out


class TestListLayerNames:
    def test_renders_table_with_purpose_and_shape(self) -> None:
        out = formatters.format_list_layer_names(
            "evo2_40b",
            [
                {"name": "blocks.20.output", "purpose": "embeddings", "shape_hint": "(n, 4096)"},
                {"name": "lm_head.output", "purpose": "logits", "shape_hint": "(n, 512)"},
            ],
        )
        assert "evo2_40b" in out
        assert "blocks.20.output" in out
        assert "embeddings" in out
        assert "lm_head.output" in out
        assert "(n, 4096)" in out


class TestNimHealth:
    def test_ready_status(self) -> None:
        out = formatters.format_nim_health(
            status="ready", base_url="http://localhost:8000"
        )
        assert "ready" in out
        assert "http://localhost:8000" in out

    def test_extra_metadata(self) -> None:
        out = formatters.format_nim_health(
            status="ready",
            base_url="http://localhost:8000",
            extra={"version": "2.1.0", "uptime_s": 1234},
        )
        assert "version" in out
        assert "2.1.0" in out
        assert "1234" in out
