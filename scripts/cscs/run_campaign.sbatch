#!/bin/bash
# CSCS Alps campaign: agent pool + vLLM node(s) + a MULTI-NODE judge pool -- the
# 3-tier agent / reference / inference topology, all in one SLURM allocation.
#
# Topology (all in one SLURM allocation):
#   node[0]        AGENT POOL -- runs AGENT_WORKERS concurrent optarena agent workers
#                               that "think" (query vLLM over HTTP). No GPU work here,
#                               so the agents never idle a timing GPU.
#   node[1..V]     vLLM       -- serves the model (tensor/pipeline parallel across V nodes).
#   node[V+1..N-1] JUDGE      -- the device pool the judge "measures" on (reference +
#                               candidate timing); each node's GPUs become judge slots.
#                               One kernel per slot; the judge work-steals across ALL
#                               judge nodes' GPUs.
#
# Both pools are driven from judge_scheduler: OPTARENA_JUDGE_NODES_EXPANDED feeds
# JudgeConfig (a slot per (judge node, GPU); remote grades via `srun --nodelist=<node>`,
# see judge_scheduler.srun_wrap), and OPTARENA_AGENT_NODES_EXPANDED /
# OPTARENA_AGENT_WORKERS_PER_NODE feed AgentPoolConfig, so TwoStageScheduler pipelines
# think (agent pool) -> grade (judge pool) with the two pools running concurrently.
#
# Submit:  sbatch -A <account> --nodes=$N scripts/cscs/run_campaign.sbatch
#   `-A <account>` is MANDATORY on Alps (or export SBATCH_ACCOUNT before submit).
# Tune via env (export before sbatch, or --export): VLLM_NODES, MODEL_ID, TP_SIZE,
#   PP_SIZE, JUDGE_GPUS_PER_NODE, AGENT_WORKERS, KERNELS, BASELINE, VLLM_PORT.
# The EDF (env.toml) selects the aarch64 (Grace-Hopper) image + CXI hook; see
# scripts/cscs/env.toml.example and docs/RUNTIME.md.
#SBATCH --job-name=optarena-campaign
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

REPO="${OPTARENA_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
ENV_TOML="${OPTARENA_ENV_TOML:-${REPO}/env.toml}"   # enroot/CXI env (see docs/RUNTIME.md)

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
VLLM_NODES="${VLLM_NODES:-1}"          # how many nodes vLLM spans (PP_SIZE across them)
VLLM_PORT="${VLLM_PORT:-8000}"
JUDGE_GPUS_PER_NODE="${JUDGE_GPUS_PER_NODE:-4}"
AGENT_WORKERS="${AGENT_WORKERS:-8}"    # concurrent agent (think) workers on the agent node
KERNELS="${KERNELS:-all}"
BASELINE="${BASELINE:-track}"          # per-track default (foundation->c-autopar, ml/hpc->numpy)
VLLM_WAIT_S="${VLLM_WAIT_S:-600}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
N="${#NODES[@]}"
(( N >= VLLM_NODES + 2 )) || { echo "need >= VLLM_NODES+2 nodes (agent + vLLM + >=1 judge); got $N" >&2; exit 2; }

AGENT_NODE="${NODES[0]}"
VLLM_HEAD="${NODES[1]}"
VLLM_HOSTS=("${NODES[@]:1:VLLM_NODES}")
JUDGE_HOSTS=("${NODES[@]:1+VLLM_NODES}")            # everything after the vLLM block
JUDGE_EXPANDED="$(IFS=,; echo "${JUDGE_HOSTS[*]}")"
VLLM_URL="http://${VLLM_HEAD}:${VLLM_PORT}/v1"

echo "agent=${AGENT_NODE}  vllm=${VLLM_HOSTS[*]} (head ${VLLM_HEAD})  judge=${JUDGE_HOSTS[*]}"
echo "vLLM URL=${VLLM_URL}  judge nodes=${#JUDGE_HOSTS[@]} x ${JUDGE_GPUS_PER_NODE} GPU"

cleanup() { jobs -p | xargs -r kill || true; wait || true; }
trap cleanup EXIT

# --- vLLM server (pipeline-parallel across the vLLM nodes) ---
for rank in "${!VLLM_HOSTS[@]}"; do
  headless=(); (( rank == 0 )) || headless=(--headless)
  srun --nodes=1 --ntasks=1 --nodelist="${VLLM_HOSTS[rank]}" --environment="${ENV_TOML}" \
    bash -lc "cd '${REPO}' && vllm serve '${MODEL_ID}' --host 0.0.0.0 --port '${VLLM_PORT}' \
      --tensor-parallel-size '${TP_SIZE}' --pipeline-parallel-size '${PP_SIZE}' \
      --distributed-executor-backend ray --node-rank '${rank}' \
      --master-addr '${VLLM_HEAD}' ${headless[*]}" &
done

# --- wait for the endpoint, then run the agent + multi-node judge on the agent node ---
srun --nodes=1 --ntasks=1 --nodelist="${AGENT_NODE}" --environment="${ENV_TOML}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    echo 'waiting for vLLM at ${VLLM_URL} ...'
    for i in \$(seq 1 $((VLLM_WAIT_S/15))); do
      curl -fsS '${VLLM_URL}/models' >/dev/null 2>&1 && { echo 'vLLM ready'; break; }
      sleep 15
    done

    # Agent pool -> vLLM over HTTP (OpenAIAgent); judge -> the judge-node GPU pool.
    # TwoStageScheduler pipelines think (agent pool) -> grade (judge pool).
    export VLLM_BASE_URL='${VLLM_URL}'
    export OPTARENA_OPENAI_MODEL='${MODEL_ID}'
    export OPTARENA_AGENT_NODES_EXPANDED='${AGENT_NODE}'
    export OPTARENA_AGENT_WORKERS_PER_NODE='${AGENT_WORKERS}'
    export OPTARENA_JUDGE_NODES_EXPANDED='${JUDGE_EXPANDED}'
    export OPTARENA_JUDGE_GPUS_PER_NODE='${JUDGE_GPUS_PER_NODE}'

    python -m optarena.cli agent openai --kernels '${KERNELS}' --baseline '${BASELINE}'
  "
