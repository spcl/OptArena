# Launching OptArena on a cluster

OptArena runs as **single-node containers** wired together by static, round-robin
assignment — no container spans nodes, nothing does dynamic load balancing, and there is
no MPI between containers. Three roles, all from the ONE universal OCI image
(`containers/optarena.Dockerfile`):

| Role | What runs in the container | How many |
|------|----------------------------|----------|
| **inference** | a vLLM server (one URL) | one per model replica |
| **judge** | `optarena serve` (the HTTP oracle: builds, times, grades) | one per judge node |
| **agent** | `optarena agent openai …` — the optimizer workers that "think" | one process, `W` workers |

An **agent worker** is bound, once and statically, to **one vLLM endpoint** (for the LLM)
and **one judge endpoint** (for the authoritative timed grade). Worker `w` uses
`vllm_urls[w % V]` and `judge_urls[w % J]`. That is the whole load-balancing story.

## Backends

Only **apptainer** and **podman** — both are what CSCS launches, and both consume the same
OCI image:

```
podman build -f containers/optarena.Dockerfile --build-arg HW=cpu -t optarena:cpu .   # OCI (add --build-arg HW=nvidia|amd)
# podman: run the tag directly.
# apptainer: build a SIF from the SAME OCI image (daemon-agnostic):
podman save optarena:cpu -o optarena-cpu.tar
apptainer build optarena-cpu.sif docker-archive:optarena-cpu.tar
```

Select the backend with `OPTARENA_RUNTIME_BACKEND=apptainer|podman` (default `apptainer`).

## Endpoints (the contract the job submission wires)

The agent reads its endpoint lists from the environment:

- `OPTARENA_VLLM_URLS` — comma-separated vLLM base URLs (e.g. `http://nid002:8000/v1,http://nid005:8000/v1`).
- `OPTARENA_JUDGE_URLS` — comma-separated judge URLs (e.g. `http://nid003:8800,http://nid006:8800`).
- `OPTARENA_AGENT_WORKERS` — number of concurrent agent workers (default: one per endpoint).

A single URL on each is fine (a small run). More than one endpoint, or `>1` worker, turns on
the distributed static path automatically (`--pipeline auto`).

## Multi-node inference (a model too big for one node)

A 4×GH200 node has ~384 GB HBM, so anything up to ~70 B dense (bf16) fits on one node;
405 B / 671 B-class models do not. For those, an inference endpoint is a **ray cluster of
single-node containers** exposing **one URL** — the ray head + workers each run in their own
single-node container and connect over the network (no container spans nodes). The agents
neither know nor care how many nodes back a URL; they just call the URL. Standing up that ray
cluster is the job submission's concern.

## Launch order

1. **Judge nodes** — start the oracle service in each judge container:
   ```
   optarena serve --host 0.0.0.0 --port 8800
   ```
2. **Inference nodes** — start vLLM in each inference container (single-node, or a ray cluster
   behind one URL for a big model).
3. **Agent** — once the judge + vLLM URLs are reachable:
   ```
   export OPTARENA_VLLM_URLS="http://nid002:8000/v1,http://nid005:8000/v1"
   export OPTARENA_JUDGE_URLS="http://nid003:8800,http://nid006:8800"
   export OPTARENA_AGENT_WORKERS=8
   optarena agent openai --kernels gemm,gesummv --baseline numpy --preset S
   ```

`--native` runs the agent + an in-process judge on one box (no containers, no endpoints) — the
serial path, for local testing.

Job submission (allocating the nodes, starting the three roles, forming any ray cluster) is
owned by the cluster's submission scripts, not by this repo.
