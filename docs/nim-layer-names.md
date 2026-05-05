# NIM `/forward` layer names

The NIM `/forward` endpoint accepts an `output_layers: array<string>` parameter
to select which intermediate tensors to return. NVIDIA's documentation does
not list valid names; this catalog is populated empirically by deploying the
container and probing it.

## Status

**Placeholder — to be populated during the Arkane H200 trial.**

The current `evo2_nim_mcp.layer_catalog` module ships with educated guesses
based on Evo2 architecture conventions (Hyena layers organized in `blocks.N`
attributes, byte-level vocab of 512). The trial will confirm or correct these.

## How to discover

Run the following from a host where the NIM is reachable:

```python
import asyncio
import base64
import io
import numpy as np
from evo2_nim_mcp.client import NimClient

CANDIDATE_NAMES = [
    "lm_head.output",
    "unembedder.output",
    "blocks.0.output",
    "blocks.20.output",
    "blocks.31.output",
    "blocks.2.mlp.l3",
    "transformer.h.0.output",
    "transformer.h.31.output",
    "embed_in.output",
]

async def probe():
    c = NimClient.from_env()
    seq = "ACGTACGTACGT"
    for name in CANDIDATE_NAMES:
        try:
            r = await c.forward({"sequence": seq, "output_layers": [name]})
            data = base64.b64decode(r["data"])
            npz = np.load(io.BytesIO(data), allow_pickle=False)
            for k in npz.files:
                print(f"  {name}: shape={npz[k].shape} dtype={npz[k].dtype}")
        except Exception as exc:
            print(f"  {name}: FAILED ({exc})")
    await c.aclose()

asyncio.run(probe())
```

For each name that succeeds:
- Note the shape (last dim 512 → LM head logits; otherwise hidden state)
- Note the dtype
- Update `evo2_nim_mcp/layer_catalog.py` with the confirmed names

## Confirmed layers

To be filled in:

| Checkpoint | Layer name | Shape | Purpose |
|---|---|---|---|
| _(empty until trial)_ | | | |

## Failed candidates

To be filled in: which names returned errors. Useful so we don't probe them again on subsequent deployments.

| Checkpoint | Tried name | Error |
|---|---|---|
| _(empty until trial)_ | | |
