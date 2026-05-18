# evo2-nim-mcp

LLM-optimized MCP server for the [NVIDIA NIM Evo2 container](https://docs.nvidia.com/nim/bionemo/evo2/latest/) — exposes the Arc Institute Evo2 DNA language model (40B or 7B) as a clean set of tools any MCP-aware client (Claude Desktop, Cursor, the MCP Inspector, custom agents) can call: scoring, batch scoring, embeddings, generation, SNP and splice variant analysis, and Ensembl-backed convenience helpers.

This is a thin HTTP wrapper around the NIM container (`nvcr.io/nim/arc/evo2:2`). It does **not** install Evo2 locally — no PyTorch, no `transformer-engine`, no `flash-attn`. It assumes the NIM container is already running on a Hopper GPU (H100 or H200) and reachable over HTTP.

## Why this exists

NVIDIA's NIM container exposes only two raw endpoints (`/biology/arc/evo2/forward` returning base64 NPZ tensors, and `/biology/arc/evo2/generate`). That is not directly LLM-callable — an LLM does not know how to decode a base64 NPZ archive, pick the right output layer, or compute a log-likelihood from raw logits. This package does that work and exposes Evo2 as twelve ergonomic MCP tools with rich docstrings telling the LLM when to use each one.

## Tool surface

| Tool | What it does |
|---|---|
| `score_sequence` | Whole-sequence log-likelihood under Evo2 |
| `score_snp` | Score a single nucleotide variant — `LL(ref)`, `LL(alt)`, delta |
| `score_variant_at` | One-call variant scoring by coordinate: fetch Ensembl context + score |
| `score_variant_batch` | Batch SNP scoring for cohort triage |
| `score_splice_region` | Score the sequence around a splice site, ref vs alt |
| `fetch_variant_context` | Pull reference DNA around a genomic coordinate (Ensembl REST) |
| `embed_sequence` | Extract embeddings at a named hidden layer |
| `embed_similarity` | Cosine similarity between two sequences (mean-pool + per-position) |
| `generate_sequence` | Conditional DNA generation |
| `list_available_checkpoints` | Which model is loaded in the NIM instance |
| `list_layer_names` | Output layers available for `/forward` (helps the LLM avoid wrong-layer errors) |
| `nim_health` | NIM readiness + version info |

Outputs are plain markdown with raw scores and minimal commentary. The wrapper deliberately does **not** map likelihood deltas to clinical categories (deleterious / benign / VUS) — Evo2's zero-shot AUROC on the BRCA1 benchmark is 0.73 ([Brixi et al 2025](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)), nowhere near a clinical classifier; interpretation is left to the caller.

## Installation

```bash
pip install git+https://github.com/zzgael/evo2-nim-mcp
```

## Configuration

Environment variables (read at process start):

| Variable | Default | Meaning |
|---|---|---|
| `EVO2_NIM_URL` | `http://localhost:8000` | URL of the NIM HTTP endpoint (loopback if MCP runs on the same host as NIM, or local end of an SSH tunnel) |
| `EVO2_NGC_API_KEY` | (unset) | Optional bearer token. Set only if your NIM deployment enforces runtime auth on the HTTP API (most don't) |
| `EVO2_REQUEST_TIMEOUT` | `90` | Per-request timeout in seconds (long enough for `/forward` on big sequences) |
| `EVO2_CONNECT_TIMEOUT` | `5` | TCP connect timeout in seconds |
| `ENSEMBL_REQUEST_TIMEOUT` | `30` | Per-request timeout (s) for the Ensembl REST sequence endpoint used by `fetch_variant_context` / `score_variant_at` |
| `ENSEMBL_CONNECT_TIMEOUT` | `5` | TCP connect timeout (s) for Ensembl |

## Running

Once a NIM container is reachable at `EVO2_NIM_URL`:

```bash
EVO2_NIM_URL=http://localhost:8000 python -m evo2_nim_mcp
```

Or via the installed entry point:

```bash
EVO2_NIM_URL=http://localhost:8000 evo2-nim-mcp
```

The server speaks the MCP stdio protocol — pipe its stdin/stdout into any MCP-compatible host.

## Using from an MCP client

### Claude Desktop

Edit your `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`, Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "evo2": {
      "command": "python",
      "args": ["-m", "evo2_nim_mcp"],
      "env": {
        "EVO2_NIM_URL": "http://localhost:8000"
      }
    }
  }
}
```

Restart Claude Desktop. The Evo2 tools will appear in the tool picker.

### Cursor

Add to `.cursor/mcp.json` in your project (or the user-level equivalent):

```json
{
  "mcpServers": {
    "evo2": {
      "command": "python",
      "args": ["-m", "evo2_nim_mcp"],
      "env": {
        "EVO2_NIM_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Any other MCP host

The server speaks the standard MCP stdio protocol. Spawn it with the right env var and connect stdin/stdout. Pseudo-config:

```
command:  python -m evo2_nim_mcp
transport: stdio
env:
  EVO2_NIM_URL: http://localhost:8000      # required
  EVO2_NGC_API_KEY: <token>                # only if NIM enforces auth
  EVO2_REQUEST_TIMEOUT: 90                 # optional, raise for very long sequences
```

Reproducibility tip: install at a pinned commit (`pip install git+https://github.com/zzgael/evo2-nim-mcp@<sha>`) rather than tracking `main` so prompts behave the same across hosts.

### Smoke test from the command line

After installing, you can hit the running NIM directly with a one-shot tools list request:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | EVO2_NIM_URL=http://localhost:8000 python -m evo2_nim_mcp
```

You should get a JSON response describing all twelve tools.

## Deploying the NIM backend

This package is the **client**. The server side is the official NVIDIA NIM container. See [`docs/deployment.md`](docs/deployment.md) for the deployment recipe (Docker run command, NGC API key, GPU requirements, SSH tunnel setup).

## License

MIT. See [LICENSE](LICENSE).
