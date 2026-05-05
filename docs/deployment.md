# Deployment

## What this MCP needs

A reachable [NVIDIA NIM Evo2 container](https://docs.nvidia.com/nim/bionemo/evo2/latest/) — typically `nvcr.io/nim/arc/evo2:2` running on a Hopper GPU (H100 80GB or H200 141GB). This package is the client; it does not host the model itself.

## Deploying the NIM container

### Hardware

| Model | GPU | VRAM |
|---|---|---|
| Evo2-40B (default) | 1× H200 141GB **or** 2× H100 80GB (NVLink) | ≥ 80 GB |
| Evo2-7B | 1× H100 80GB / H200 141GB / RTX 6000 Ada / L40S | ≥ 24 GB |

System: x86_64 Linux, NVIDIA driver ≥ 550.90.07, Docker ≥ 23.0.1, NVIDIA Container Toolkit ≥ 1.13.5, 110 GB of disk for the cached weights, 16 GB CPU RAM.

### Auth

You need an [NGC API key](https://ngc.nvidia.com) (free signup). Set it as `NGC_API_KEY` in the container's environment.

### Run command

```bash
export NGC_API_KEY=<your-NGC-API-key>
export LOCAL_NIM_CACHE=/var/lib/nim-cache  # persistent storage, NOT /tmp

mkdir -p "$LOCAL_NIM_CACHE"

docker run -d --restart unless-stopped \
    --name evo2-40b \
    --runtime=nvidia \
    --gpus='"device=0"' \
    -p 8000:8000 \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE":/opt/nim/.cache \
    nvcr.io/nim/arc/evo2:2
```

To run the 7B model instead, add `-e NIM_VARIANT=7b`.

The first start downloads ~30 GB of weights from NGC (~10–30 min depending on network); subsequent starts hit the cache and ready in 2–5 min. Confirm readiness with:

```bash
curl http://localhost:8000/v1/health/ready
# {"status":"ready"}
```

## Connecting this MCP

### Direct (NIM and MCP on the same host)

```bash
pip install git+https://github.com/zzgael/evo2-nim-mcp
EVO2_NIM_URL=http://localhost:8000 python -m evo2_nim_mcp
```

### Through SSH tunnel (NIM on remote H200, MCP on your laptop / Workbench prod)

```bash
ssh -L 8000:localhost:8000 -N <user>@<gpu-host> &
EVO2_NIM_URL=http://localhost:8000 python -m evo2_nim_mcp
```

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `EVO2_NIM_URL` | `http://localhost:8000` | URL of the NIM HTTP endpoint (loopback if MCP is on the same host as NIM, or local end of an SSH tunnel) |
| `EVO2_NGC_API_KEY` | (unset) | Bearer token, only set if your NIM requires runtime auth (most don't) |
| `EVO2_REQUEST_TIMEOUT` | `90` | Per-request timeout in seconds |
| `EVO2_CONNECT_TIMEOUT` | `5` | TCP connect timeout in seconds |
| `NIM_VARIANT` | `40b` | Read by `list_available_checkpoints` to label the loaded model. Should match the env var on the actual container. |

## Workbench MCP config snippet

In a GPT Workbench-style healthcare integration:

```typescript
import { McpToolConfig } from '...';
import { buildPythonPackageCheck } from '...';

export const EVO2_MCP_CONFIG: McpToolConfig = {
    command: 'bash',
    args: [
        '-c',
        `${buildPythonPackageCheck(['evo2_nim_mcp'])} || ` +
            'pip3 install --user --quiet --break-system-packages ' +
            'git+https://github.com/zzgael/evo2-nim-mcp@<commit-sha> 2>/dev/null; ' +
            'exec python3 -m evo2_nim_mcp',
    ],
    transport: 'stdio',
    envMappings: [
        { serverEnvVarName: 'EVO2_NIM_URL', valueSource: { type: 'SYSTEM_CONFIG', path: 'app.evo2.nimUrl' } },
    ],
    // ...
};
```

Pin `<commit-sha>` to a known-good commit so deployments are reproducible.

## Verifying it works

```bash
# After starting the MCP, in another terminal:
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | python -m evo2_nim_mcp
```

Or use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) for an interactive UI.
