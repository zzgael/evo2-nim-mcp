"""Unit tests for evo2_nim_mcp.client. All HTTP interactions mocked with respx."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from evo2_nim_mcp.client import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_NIM_URL,
    DEFAULT_REQUEST_TIMEOUT,
    NimClient,
    NimError,
    NimNotReadyError,
)


@pytest.fixture
async def client() -> NimClient:
    c = NimClient(base_url="http://nim.test")
    yield c
    await c.aclose()


class TestFromEnv:
    def test_defaults_when_env_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            c = NimClient.from_env()
            assert c.base_url == DEFAULT_NIM_URL
            assert c.request_timeout == DEFAULT_REQUEST_TIMEOUT
            assert c.connect_timeout == DEFAULT_CONNECT_TIMEOUT

    def test_reads_url_from_env(self) -> None:
        with patch.dict(os.environ, {"EVO2_NIM_URL": "http://h200.example:9999"}, clear=True):
            c = NimClient.from_env()
            assert c.base_url == "http://h200.example:9999"

    def test_strips_trailing_slash_from_url(self) -> None:
        c = NimClient(base_url="http://nim.test/")
        assert c.base_url == "http://nim.test"

    def test_reads_timeouts_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"EVO2_REQUEST_TIMEOUT": "120", "EVO2_CONNECT_TIMEOUT": "10"},
            clear=True,
        ):
            c = NimClient.from_env()
            assert c.request_timeout == 120.0
            assert c.connect_timeout == 10.0

    def test_invalid_timeout_falls_back_to_default(self) -> None:
        with patch.dict(os.environ, {"EVO2_REQUEST_TIMEOUT": "not-a-number"}, clear=True):
            c = NimClient.from_env()
            assert c.request_timeout == DEFAULT_REQUEST_TIMEOUT

    def test_ngc_api_key_sets_authorization_header(self) -> None:
        with patch.dict(os.environ, {"EVO2_NGC_API_KEY": "nvapi-secret123"}, clear=True):
            c = NimClient.from_env()
            assert c._client.headers["Authorization"] == "Bearer nvapi-secret123"

    def test_no_ngc_api_key_means_no_authorization_header(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            c = NimClient.from_env()
            assert "Authorization" not in c._client.headers


class TestHealth:
    @respx.mock
    async def test_ready_returns_body(self, client: NimClient) -> None:
        respx.get("http://nim.test/v1/health/ready").mock(
            return_value=httpx.Response(200, json={"status": "ready"})
        )
        result = await client.health()
        assert result == {"status": "ready"}

    @respx.mock
    async def test_not_ready_raises_typed_error(self, client: NimClient) -> None:
        respx.get("http://nim.test/v1/health/ready").mock(
            return_value=httpx.Response(200, json={"status": "loading"})
        )
        with pytest.raises(NimNotReadyError) as excinfo:
            await client.health()
        assert "loading" in str(excinfo.value)
        assert "30 seconds" in str(excinfo.value)

    @respx.mock
    async def test_503_retries_then_fails_with_helpful_message(self, client: NimClient) -> None:
        respx.get("http://nim.test/v1/health/ready").mock(
            return_value=httpx.Response(503, json={"detail": "loading"})
        )
        with pytest.raises(NimError) as excinfo:
            await client.health()
        assert "503" in str(excinfo.value)


class TestGenerate:
    @respx.mock
    async def test_success_returns_json(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/generate").mock(
            return_value=httpx.Response(
                200,
                json={"sequence": "ACGT", "elapsed_ms": 100},
            )
        )
        result = await client.generate({"sequence": "AC", "num_tokens": 2})
        assert result == {"sequence": "ACGT", "elapsed_ms": 100}

    @respx.mock
    async def test_4xx_surfaces_immediately_with_detail(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/generate").mock(
            return_value=httpx.Response(
                400,
                json={"detail": "Invalid temperature: must be > 0"},
            )
        )
        with pytest.raises(NimError) as excinfo:
            await client.generate({"sequence": "AC", "temperature": -1})
        msg = str(excinfo.value)
        assert "400" in msg
        assert "Invalid temperature" in msg

    @respx.mock
    async def test_4xx_does_not_retry(self, client: NimClient) -> None:
        route = respx.post("http://nim.test/biology/arc/evo2/generate").mock(
            return_value=httpx.Response(404, json={"detail": "endpoint not found"})
        )
        with pytest.raises(NimError):
            await client.generate({"sequence": "AC"})
        assert route.call_count == 1  # No retry on 4xx


class TestForward:
    @respx.mock
    async def test_success_returns_data_field(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/forward").mock(
            return_value=httpx.Response(
                200,
                json={"data": "base64encodedNPZhere", "elapsed_ms": 234},
            )
        )
        result = await client.forward(
            {"sequence": "ACGTACGT", "output_layers": ["lm_head.output"]}
        )
        assert result["data"] == "base64encodedNPZhere"
        assert result["elapsed_ms"] == 234

    @respx.mock
    async def test_5xx_retries_with_backoff(self, client: NimClient) -> None:
        # Two 503s then a 200 on the third try
        responses = [
            httpx.Response(503, json={"detail": "overloaded"}),
            httpx.Response(503, json={"detail": "overloaded"}),
            httpx.Response(200, json={"data": "ok", "elapsed_ms": 50}),
        ]
        route = respx.post("http://nim.test/biology/arc/evo2/forward").mock(side_effect=responses)
        result = await client.forward({"sequence": "AC", "output_layers": ["x"]})
        assert result == {"data": "ok", "elapsed_ms": 50}
        assert route.call_count == 3

    @respx.mock
    async def test_persistent_5xx_eventually_raises(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/forward").mock(
            return_value=httpx.Response(500, json={"detail": "server crashed"})
        )
        with pytest.raises(NimError) as excinfo:
            await client.forward({"sequence": "AC", "output_layers": ["x"]})
        assert "500" in str(excinfo.value)
        assert "server crashed" in str(excinfo.value)


class TestNetworkErrors:
    @respx.mock
    async def test_connect_error_retries_then_raises_with_tunnel_hint(
        self, client: NimClient
    ) -> None:
        respx.post("http://nim.test/biology/arc/evo2/forward").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(NimError) as excinfo:
            await client.forward({"sequence": "AC", "output_layers": ["x"]})
        msg = str(excinfo.value)
        assert "unreachable" in msg.lower()
        assert "tunnel" in msg.lower()

    @respx.mock
    async def test_timeout_retries_then_raises(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/forward").mock(
            side_effect=httpx.TimeoutException("read timeout")
        )
        with pytest.raises(NimError):
            await client.forward({"sequence": "AC", "output_layers": ["x"]})


class TestNonJsonResponses:
    @respx.mock
    async def test_html_response_raises_clear_error(self, client: NimClient) -> None:
        respx.get("http://nim.test/v1/health/ready").mock(
            return_value=httpx.Response(
                200,
                content=b"<html>oops</html>",
                headers={"Content-Type": "text/html"},
            )
        )
        with pytest.raises(NimError) as excinfo:
            await client.health()
        assert "non-JSON" in str(excinfo.value)


class TestErrorDetailExtraction:
    @respx.mock
    async def test_extracts_error_field(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/generate").mock(
            return_value=httpx.Response(400, json={"error": "validation failed"})
        )
        with pytest.raises(NimError) as excinfo:
            await client.generate({})
        assert "validation failed" in str(excinfo.value)

    @respx.mock
    async def test_falls_back_to_text_for_non_json_4xx(self, client: NimClient) -> None:
        respx.post("http://nim.test/biology/arc/evo2/generate").mock(
            return_value=httpx.Response(400, content=b"plain text error")
        )
        with pytest.raises(NimError) as excinfo:
            await client.generate({})
        assert "plain text error" in str(excinfo.value)
