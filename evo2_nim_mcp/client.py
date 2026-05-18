"""Async HTTP client for the NVIDIA NIM Evo2 container.

The NIM speaks REST on port 8000 of the host where it runs. This module wraps
that HTTP surface in a single shared `httpx.AsyncClient` with sensible defaults,
retries, and clear error messages.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_NIM_URL = "http://localhost:8000"
DEFAULT_REQUEST_TIMEOUT = 90.0
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_MAX_CONCURRENT = 1  # NIM serves one inference at a time; bursting wastes attempts.
DEFAULT_RETRIES = 50  # Generous: backpressure can stack across users on the shared NIM.
HEALTH_PATH = "/v1/health/ready"
GENERATE_PATH = "/biology/arc/evo2/generate"
FORWARD_PATH = "/biology/arc/evo2/forward"


class NimError(Exception):
    """Raised when the NIM returns a non-success response or is unreachable.

    The message is formatted for LLM consumption — include the failing operation,
    the offending parameter (if known), and a hint when applicable.
    """


class NimNotReadyError(NimError):
    """NIM container is up but the model has not finished loading yet."""


@dataclass
class NimResponse:
    """A successful NIM response, decoupled from httpx for easier testing."""

    status_code: int
    json: dict[str, Any]


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class NimClient:
    """Single-instance NIM HTTP client.

    Usage:
        client = NimClient.from_env()
        await client.health()
        await client.generate(sequence="ACGT", num_tokens=10)
        await client.aclose()
    """

    def __init__(
        self,
        base_url: str = DEFAULT_NIM_URL,
        *,
        ngc_api_key: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.connect_timeout = connect_timeout
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if ngc_api_key:
            headers["Authorization"] = f"Bearer {ngc_api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            headers=headers,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        # In-process serialization. NIM has 1 HTTP worker; bursting N parallel
        # requests from a single AIMessage just causes HTTP 422 "Too Busy"
        # rejections that we then have to retry. Serialize at the source.
        # This does NOT help across MCP processes (one per chat run) — the
        # retry-with-backoff loop is the safety net for that case.
        self._inflight = asyncio.Semaphore(max(1, max_concurrent))

    @classmethod
    def from_env(cls) -> NimClient:
        """Construct a client from environment variables.

        Reads:
        - EVO2_NIM_URL (default http://localhost:8000)
        - EVO2_NGC_API_KEY (optional)
        - EVO2_REQUEST_TIMEOUT (default 90s)
        - EVO2_CONNECT_TIMEOUT (default 5s)
        - EVO2_MAX_CONCURRENT (default 1 — see __init__ docstring)
        """

        def _i(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except ValueError:
                return default

        return cls(
            base_url=os.environ.get("EVO2_NIM_URL", DEFAULT_NIM_URL),
            ngc_api_key=os.environ.get("EVO2_NGC_API_KEY") or None,
            request_timeout=_read_env_float("EVO2_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
            connect_timeout=_read_env_float("EVO2_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT),
            max_concurrent=_i("EVO2_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API — one method per NIM endpoint
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """GET /v1/health/ready — returns the JSON body on 200, raises NimNotReadyError otherwise."""
        resp = await self._get(HEALTH_PATH)
        body = resp.json
        if body.get("status") != "ready":
            raise NimNotReadyError(
                f"NIM is not ready yet. status={body.get('status')!r}. "
                "The container has likely not finished loading the model — try again in 30 seconds."
            )
        return body

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /biology/arc/evo2/generate.

        Caller is responsible for shaping `payload` to match the NIM's parameter schema:
        sequence, num_tokens, temperature, top_k, top_p, random_seed,
        enable_logits, enable_sampled_probs, enable_elapsed_ms_per_token.
        """
        return (await self._post(GENERATE_PATH, payload)).json

    async def forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /biology/arc/evo2/forward.

        Caller is responsible for shaping `payload`: sequence, output_layers.
        Response: {"data": "<base64 NPZ>", "elapsed_ms": int}
        """
        return (await self._post(FORWARD_PATH, payload)).json

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, *, retries: int = DEFAULT_RETRIES) -> NimResponse:
        return await self._request("GET", path, json=None, retries=retries)

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        retries: int = DEFAULT_RETRIES,
    ) -> NimResponse:
        return await self._request("POST", path, json=payload, retries=retries)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None,
        retries: int,
    ) -> NimResponse:
        """One request with exponential backoff on:
          - 5xx errors
          - transient network errors
          - HTTP 422 "Too Busy" (NIM backpressure under shared load)

        Other 4xx are caller errors and surface immediately.

        All requests are serialized through `self._inflight` so we don't fight
        ourselves for the GPU within a single MCP process.
        """
        delay = 1.0
        BACKOFF_CAP = 30.0
        last_error: Exception | None = None
        async with self._inflight:
            for attempt in range(retries + 1):
                try:
                    resp = await self._client.request(method, path, json=json)
                except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as exc:
                    last_error = exc
                    if attempt == retries:
                        raise NimError(
                            f"NIM unreachable at {self.base_url}{path}: {type(exc).__name__}: {exc}. "
                            "Check that the NIM container is running and the SSH tunnel is up."
                        ) from exc
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, BACKOFF_CAP)
                    continue

                if resp.status_code < 400:
                    try:
                        return NimResponse(status_code=resp.status_code, json=resp.json())
                    except ValueError as exc:
                        raise NimError(
                            f"NIM returned non-JSON body for {method} {path} "
                            f"(status {resp.status_code}): {resp.text[:200]!r}"
                        ) from exc

                if 400 <= resp.status_code < 500:
                    detail = self._extract_error_detail(resp)
                    # NIM uses HTTP 422 for BOTH input validation errors AND
                    # backpressure ("Too Busy"). The latter is transient — retry.
                    # Other 422s and 4xx are caller errors and should fail fast.
                    if resp.status_code == 422 and "too busy" in detail.lower():
                        last_error = NimError(
                            f"NIM backpressure (HTTP 422 Too Busy) for {method} {path}"
                        )
                        if attempt == retries:
                            raise last_error
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, BACKOFF_CAP)
                        continue
                    raise NimError(
                        f"NIM rejected {method} {path} with HTTP {resp.status_code}: {detail}"
                    )

                # 5xx — retry with backoff
                last_error = NimError(
                    f"NIM returned HTTP {resp.status_code} for {method} {path}: "
                    f"{self._extract_error_detail(resp)}"
                )
                if attempt == retries:
                    raise last_error
                await asyncio.sleep(delay)
                delay = min(delay * 2, BACKOFF_CAP)

        # Unreachable in practice; satisfies type checker
        raise last_error or NimError(f"NIM request failed: {method} {path}")

    @staticmethod
    def _extract_error_detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict):
                return str(body.get("detail") or body.get("error") or body)
            return str(body)
        except ValueError:
            return resp.text[:200] or "<empty body>"
