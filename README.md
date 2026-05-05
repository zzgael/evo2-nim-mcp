# evo2-nim-mcp

LLM-optimized MCP server for the [NVIDIA NIM Evo2 container](https://docs.nvidia.com/nim/bionemo/evo2/latest/) — exposes the Arc Institute Evo2 DNA language model (40B or 7B) as a clean set of tools an LLM can call: scoring, batch scoring, embeddings, generation, SNP and splice variant analysis.

This is a thin HTTP wrapper around the NIM container (`nvcr.io/nim/arc/evo2:2`). It does **not** install Evo2 locally — no PyTorch, no `transformer-engine`, no `flash-attn`. It assumes the NIM container is already running on a Hopper GPU (H100 or H200) and reachable over HTTP.

## Why this exists

NVIDIA's NIM container exposes only two raw endpoints (`/biology/arc/evo2/forward` returning base64 NPZ tensors, and `/biology/arc/evo2/generate`). That is not directly LLM-callable — an LLM does not know how to decode a base64 NPZ archive, pick the right output layer, or compute a log-likelihood from raw logits. This package does that work and exposes Evo2 as nine ergonomic MCP tools with rich docstrings telling the LLM when to use each one.

## Tool surface

| Tool | What it does |
|---|---|
| `score_sequence` | Whole-sequence likelihood under Evo2 |
| `score_snp` | Pathogenicity score for a single nucleotide variant (ref vs alt + delta) |
| `score_variant_batch` | Batch SNP scoring for cohort triage |
| `score_splice_region` | Score sequence around a splice site, ref vs alt |
| `embed_sequence` | Extract embeddings at a named hidden layer |
| `generate_sequence` | Conditional DNA generation |
| `list_available_checkpoints` | Which model is loaded in the NIM instance |
| `list_layer_names` | Output layers available for `/forward` (helps the LLM avoid wrong-layer errors) |
| `nim_health` | NIM readiness + version info |

Every tool returns LLM-friendly markdown with structured headings, runtime breakdown, cache hit transparency, and interpretation hints.

## Installation

```bash
pip install git+https://github.com/zzgael/evo2-nim-mcp
```

## Configuration

Environment variables (read at process start):

| Variable | Default | Meaning |
|---|---|---|
| `EVO2_NIM_URL` | `http://localhost:8000` | Local end of an SSH tunnel forwarding the NIM container's port |
| `EVO2_NGC_API_KEY` | (unset) | Optional. Some NIM deployments require runtime auth |
| `EVO2_REQUEST_TIMEOUT` | `90` | Per-request timeout in seconds (long enough for `/forward` on big sequences) |
| `EVO2_CONNECT_TIMEOUT` | `5` | TCP connect timeout in seconds |

## Running

Once a NIM container is reachable at `EVO2_NIM_URL`:

```bash
EVO2_NIM_URL=http://localhost:8000 python -m evo2_nim_mcp
```

Or via the installed entry point:

```bash
EVO2_NIM_URL=http://localhost:8000 evo2-nim-mcp
```

The server speaks the MCP stdio protocol — wire it into Claude Desktop, Cursor, GPT Workbench, or any MCP-compatible host.

## Deploying the NIM backend

This package is the **client**. The server side is the official NVIDIA NIM container. See [`docs/deployment.md`](docs/deployment.md) for the deployment recipe (Docker run command, NGC API key, GPU requirements).

## License

MIT. See [LICENSE](LICENSE).
