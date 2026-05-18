"""Ensembl REST API client for fetching reference genome sequence.

The Evo2 NIM `/forward` and `/generate` endpoints need raw ACGT context to score
or generate from. The VEP MCP returns variant annotations but does not return
the surrounding reference DNA, so we fetch it ourselves from Ensembl's public
sequence endpoint.

This module is intentionally scoped to "fetch DNA around a coordinate" — not a
general Ensembl client. If we need more later (transcripts, regulatory
elements, etc.) it belongs in a dedicated Ensembl MCP, not here.

Reference: https://rest.ensembl.org/documentation/info/sequence_region
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Final

import httpx

# Ensembl REST host varies by genome assembly. GRCh38 is the default; GRCh37
# (hg19) lives on a separate subdomain. We hardcode the canonical hosts; other
# assemblies are not supported.
_HOSTS: Final[dict[str, str]] = {
    "GRCh38": "https://rest.ensembl.org",
    "GRCh37": "https://grch37.rest.ensembl.org",
}

# Ensembl publishes a 15 req/sec rate limit. We stay comfortably under it.
_MIN_INTERVAL_S: Final[float] = 0.08

_VALID_BASES: Final[set[str]] = {"A", "C", "G", "T", "N"}


class EnsemblError(Exception):
    """Raised when Ensembl REST returns an error or unexpected response."""


@dataclass
class FetchedContext:
    """Result of fetching reference DNA around a variant position.

    Attributes:
        sequence: The fetched DNA, uppercase ACGT(N), with no whitespace.
        chromosome: Chromosome name as queried (without `chr` prefix).
        start: 1-based inclusive start coordinate of the returned sequence.
        end: 1-based inclusive end coordinate.
        center_index: 0-based index inside `sequence` corresponding to the
            variant position the caller asked for. `sequence[center_index]`
            is the reference base at that coordinate.
        assembly: Genome assembly used (e.g. "GRCh38").
        species: Species used (e.g. "human").
    """

    sequence: str
    chromosome: str
    start: int
    end: int
    center_index: int
    assembly: str
    species: str


class EnsemblClient:
    """Thin async client around the Ensembl REST `sequence/region` endpoint.

    One shared `httpx.AsyncClient` per process. A simple monotonic-clock guard
    enforces ~80 ms between requests so we stay under the 15 req/sec limit
    Ensembl publishes for unauthenticated traffic.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 30.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            headers={
                "User-Agent": "evo2-nim-mcp/ensembl-helper (gpt-workbench)",
                "Accept": "text/plain",
            },
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            follow_redirects=True,
        )
        self._lock = asyncio.Lock()
        self._next_allowed_at: float = 0.0

    @classmethod
    def from_env(cls) -> EnsemblClient:
        """Construct with timeouts driven by env vars (same names as NIM client where applicable).

        - `ENSEMBL_REQUEST_TIMEOUT` (default 30s)
        - `ENSEMBL_CONNECT_TIMEOUT` (default 5s)
        """

        def _f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except ValueError:
                return default

        return cls(
            request_timeout=_f("ENSEMBL_REQUEST_TIMEOUT", 30.0),
            connect_timeout=_f("ENSEMBL_CONNECT_TIMEOUT", 5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_region(
        self,
        chromosome: str,
        start: int,
        end: int,
        *,
        species: str = "human",
        assembly: str = "GRCh38",
    ) -> str:
        """Fetch raw DNA for `chromosome:start..end` (1-based inclusive).

        Returns the sequence as uppercase ACGT(N) with whitespace stripped.
        Raises `EnsemblError` on HTTP failure, unknown assembly, or
        unexpectedly empty response.
        """
        if assembly not in _HOSTS:
            raise EnsemblError(
                f"Unsupported assembly {assembly!r}. Supported: {sorted(_HOSTS)}."
            )
        if start < 1 or end < start:
            raise EnsemblError(
                f"Invalid region: start={start}, end={end}. Need 1 <= start <= end."
            )

        # Strip optional `chr` prefix (some clients pass "chr17", Ensembl wants "17").
        chrom = chromosome.removeprefix("chr").removeprefix("CHR")

        url = (
            f"{_HOSTS[assembly]}/sequence/region/{species}/"
            f"{chrom}:{start}..{end}"
            "?content-type=text/plain"
        )

        await self._rate_limit_wait()

        try:
            r = await self._client.get(url)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise EnsemblError(
                f"Ensembl unreachable at {url}: {type(exc).__name__}: {exc}."
            ) from exc

        if r.status_code == 400:
            raise EnsemblError(
                f"Ensembl rejected region {chrom}:{start}..{end} on {assembly} "
                f"(HTTP 400): {r.text[:200]}. Check chromosome name and coordinates."
            )
        if r.status_code == 404:
            raise EnsemblError(
                f"Ensembl returned 404 for {chrom}:{start}..{end} on {assembly}. "
                "Likely an invalid region or chromosome name."
            )
        if r.status_code >= 500:
            raise EnsemblError(
                f"Ensembl returned HTTP {r.status_code} for {url}: {r.text[:200]}"
            )
        if r.status_code != 200:
            raise EnsemblError(
                f"Ensembl returned unexpected HTTP {r.status_code} for {url}: "
                f"{r.text[:200]}"
            )

        seq = "".join(r.text.split()).upper()
        if not seq:
            raise EnsemblError(
                f"Ensembl returned an empty sequence for {chrom}:{start}..{end} on {assembly}."
            )
        if any(c not in _VALID_BASES for c in seq):
            unknown = sorted({c for c in seq if c not in _VALID_BASES})[:5]
            raise EnsemblError(
                f"Ensembl response contains unexpected characters {unknown} "
                f"for {chrom}:{start}..{end} on {assembly}."
            )
        return seq

    async def fetch_variant_context(
        self,
        chromosome: str,
        position: int,
        *,
        window_size: int = 8192,
        species: str = "human",
        assembly: str = "GRCh38",
    ) -> FetchedContext:
        """Fetch reference DNA centred on `chromosome:position` (1-based).

        The returned window is exactly `window_size` bp long when the position
        is far from the chromosome ends; it may be shorter if `position` is
        within `window_size // 2` of a telomere.

        Default `window_size=8192` matches the Arc Institute BRCA1 zero-shot
        notebook (`WINDOW_SIZE = 8192`), the published methodology for
        Evo2 SNP scoring.

        Raises `EnsemblError` on network/HTTP failures or `ValueError` on bad
        input.
        """
        if window_size < 2 or window_size > 10_000:
            raise ValueError(
                f"window_size must be 2..10000 (NIM /forward hard cap); got {window_size}."
            )
        if position < 1:
            raise ValueError(f"position must be 1-based positive; got {position}.")

        half = window_size // 2
        start = max(1, position - half)
        end = position + (window_size - half) - 1
        seq = await self.fetch_region(
            chromosome, start, end, species=species, assembly=assembly
        )
        center_index = position - start

        return FetchedContext(
            sequence=seq,
            chromosome=chromosome.removeprefix("chr").removeprefix("CHR"),
            start=start,
            end=start + len(seq) - 1,
            center_index=center_index,
            assembly=assembly,
            species=species,
        )

    async def _rate_limit_wait(self) -> None:
        """Block until ~80 ms have passed since the last outgoing request."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed_at - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed_at = loop.time() + _MIN_INTERVAL_S
