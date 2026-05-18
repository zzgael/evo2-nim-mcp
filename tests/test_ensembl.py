"""Unit tests for evo2_nim_mcp.ensembl.

These tests stub the HTTP layer with `respx` so they don't hit the real
Ensembl REST API. End-to-end coverage of the live API lives in the
integration smoke test in `docs/nim-layer-names.md` style probes.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from evo2_nim_mcp import ensembl


class TestFetchRegion:
    @pytest.mark.asyncio
    async def test_returns_uppercase_acgt_stripping_whitespace(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            mock.get(
                "/sequence/region/human/17:41276131..41276136"
            ).mock(return_value=httpx.Response(200, text="acgtnA\nC G\n"))
            client = ensembl.EnsemblClient()
            seq = await client.fetch_region("17", 41276131, 41276136)
            assert seq == "ACGTNACG"
            await client.aclose()

    @pytest.mark.asyncio
    async def test_strips_chr_prefix(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            mock.get(
                "/sequence/region/human/16:2138711..2138715"
            ).mock(return_value=httpx.Response(200, text="ACGTN"))
            client = ensembl.EnsemblClient()
            seq = await client.fetch_region("chr16", 2138711, 2138715)
            assert seq == "ACGTN"
            await client.aclose()

    @pytest.mark.asyncio
    async def test_grch37_routes_to_separate_host(self) -> None:
        async with respx.mock(base_url="https://grch37.rest.ensembl.org") as mock:
            mock.get("/sequence/region/human/17:1..3").mock(
                return_value=httpx.Response(200, text="ACG")
            )
            client = ensembl.EnsemblClient()
            seq = await client.fetch_region("17", 1, 3, assembly="GRCh37")
            assert seq == "ACG"
            await client.aclose()

    @pytest.mark.asyncio
    async def test_unknown_assembly_raises(self) -> None:
        client = ensembl.EnsemblClient()
        with pytest.raises(ensembl.EnsemblError, match="Unsupported assembly"):
            await client.fetch_region("17", 1, 3, assembly="hg18")
        await client.aclose()

    @pytest.mark.asyncio
    async def test_invalid_region_raises(self) -> None:
        client = ensembl.EnsemblClient()
        with pytest.raises(ensembl.EnsemblError, match="Invalid region"):
            await client.fetch_region("17", 100, 50)
        with pytest.raises(ensembl.EnsemblError, match="Invalid region"):
            await client.fetch_region("17", 0, 10)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_404_raises_with_clear_message(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            mock.get("/sequence/region/human/ZZ:1..3").mock(
                return_value=httpx.Response(404, text="region not found")
            )
            client = ensembl.EnsemblClient()
            with pytest.raises(ensembl.EnsemblError, match="404"):
                await client.fetch_region("ZZ", 1, 3)
            await client.aclose()

    @pytest.mark.asyncio
    async def test_empty_response_raises(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            mock.get("/sequence/region/human/17:1..3").mock(
                return_value=httpx.Response(200, text="")
            )
            client = ensembl.EnsemblClient()
            with pytest.raises(ensembl.EnsemblError, match="empty"):
                await client.fetch_region("17", 1, 3)
            await client.aclose()

    @pytest.mark.asyncio
    async def test_unexpected_chars_raise(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            mock.get("/sequence/region/human/17:1..5").mock(
                return_value=httpx.Response(200, text="ACGTX")
            )
            client = ensembl.EnsemblClient()
            with pytest.raises(ensembl.EnsemblError, match="unexpected characters"):
                await client.fetch_region("17", 1, 5)
            await client.aclose()


class TestFetchVariantContext:
    @pytest.mark.asyncio
    async def test_centers_window_on_position(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            # For position=1000, window_size=10: half=5, start=1000-5=995, end=1000+(10-5)-1=1004
            # window includes positions 995..1004 (10 bp)
            mock.get(
                "/sequence/region/human/17:995..1004"
            ).mock(return_value=httpx.Response(200, text="AAAAACGTTT"))
            client = ensembl.EnsemblClient()
            ctx = await client.fetch_variant_context("17", 1000, window_size=10)
            assert ctx.sequence == "AAAAACGTTT"
            assert ctx.center_index == 5  # position 1000 is at index 1000-995=5
            assert ctx.sequence[ctx.center_index] == "C"
            assert ctx.start == 995
            assert ctx.end == 1004
            assert ctx.chromosome == "17"
            await client.aclose()

    @pytest.mark.asyncio
    async def test_clamps_start_at_telomere(self) -> None:
        async with respx.mock(base_url="https://rest.ensembl.org") as mock:
            # position=3, window_size=10 → would want start=3-5=-2, clamped to 1
            mock.get("/sequence/region/human/17:1..7").mock(
                return_value=httpx.Response(200, text="ACGTNAC")
            )
            client = ensembl.EnsemblClient()
            ctx = await client.fetch_variant_context("17", 3, window_size=10)
            assert ctx.start == 1
            assert ctx.center_index == 2  # position 3 at index 3-1=2
            assert ctx.sequence[ctx.center_index] == "G"
            await client.aclose()

    @pytest.mark.asyncio
    async def test_rejects_oversize_window(self) -> None:
        client = ensembl.EnsemblClient()
        with pytest.raises(ValueError, match="window_size"):
            await client.fetch_variant_context("17", 1000, window_size=100_000)
        with pytest.raises(ValueError, match="window_size"):
            await client.fetch_variant_context("17", 1000, window_size=1)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_rejects_invalid_position(self) -> None:
        client = ensembl.EnsemblClient()
        with pytest.raises(ValueError, match="position must be 1-based"):
            await client.fetch_variant_context("17", 0)
        with pytest.raises(ValueError, match="position must be 1-based"):
            await client.fetch_variant_context("17", -10)
        await client.aclose()
