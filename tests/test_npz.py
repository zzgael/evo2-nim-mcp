"""Unit tests for evo2_nim_mcp.npz."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from evo2_nim_mcp.npz import NpzDecodeError, decode_forward_response


class TestDecodeForwardResponse:
    def test_decodes_single_array(self, random_logits_b64: str) -> None:
        arrays = decode_forward_response(random_logits_b64)
        assert "lm_head.output" in arrays
        assert arrays["lm_head.output"].shape == (12, 512)
        assert arrays["lm_head.output"].dtype == np.float32

    def test_decodes_multiple_arrays(self) -> None:
        from tests.conftest import npz_to_b64

        b64 = npz_to_b64(
            {
                "layer_a": np.zeros((4, 8), dtype=np.float32),
                "layer_b": np.ones((4, 16), dtype=np.float32),
            }
        )
        arrays = decode_forward_response(b64)
        assert set(arrays) == {"layer_a", "layer_b"}
        assert arrays["layer_a"].shape == (4, 8)
        assert arrays["layer_b"].shape == (4, 16)
        np.testing.assert_array_equal(arrays["layer_b"], 1.0)

    def test_empty_data_raises_with_layer_hint(self) -> None:
        with pytest.raises(NpzDecodeError) as excinfo:
            decode_forward_response("")
        msg = str(excinfo.value)
        assert "empty" in msg.lower()
        assert "list_layer_names" in msg

    def test_invalid_base64_raises(self) -> None:
        with pytest.raises(NpzDecodeError) as excinfo:
            decode_forward_response("not-valid-base64!@#")
        assert "non-base64" in str(excinfo.value)

    def test_malformed_npz_raises(self) -> None:
        # Valid base64 but not a NPZ archive
        garbage = base64.b64encode(b"this is not an NPZ").decode("ascii")
        with pytest.raises(NpzDecodeError) as excinfo:
            decode_forward_response(garbage)
        assert "not a valid NPZ archive" in str(excinfo.value)

    def test_pickled_npz_rejected_for_security(self) -> None:
        # NPZ files can contain pickled objects when allow_pickle=True; we set False.
        # Construct a payload that requires allow_pickle and verify decode rejects it.
        buf = io.BytesIO()
        np.savez(buf, evil=np.array(["string"], dtype=object))  # object dtype forces pickle
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        with pytest.raises(NpzDecodeError):
            decode_forward_response(b64)

    def test_returns_dict_keyed_by_layer_name(self) -> None:
        from tests.conftest import npz_to_b64

        b64 = npz_to_b64({"transformer.h.31.output": np.zeros((3, 4096), dtype=np.float32)})
        arrays = decode_forward_response(b64)
        assert list(arrays.keys()) == ["transformer.h.31.output"]
