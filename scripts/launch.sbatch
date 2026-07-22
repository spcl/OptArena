#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# One SLURM job -> the whole static HPCAgent-Bench deployment on a homogeneous GH200 cluster
# (Daint/Alps: every node is 4x GH200). `hpcagent-bench launch` runs under one srun across the
# WHOLE allocation, one task per node; MPI gives each rank a node and its rank decides its
# role (see hpcagent_bench/harness/cluster_launch.py):
#
#   ranks [0, I*K)          -- inference: groups of K nodes form one vLLM endpoint
#   ranks [I*K, I*K + J)    -- judge: one `hpcagent-bench serve` each
#   rank 0                  -- ALSO the agent driver (co-located; HTTP-only, GPU-idle)
#
# so the allocation must be exactly  N = inference_endpoints*nodes_per_vllm + judge_nodes.
# Example below: I=2 single-node endpoints (K=1) + J=1 judge = 3 nodes.
#
#   sbatch scripts/launch.sbatch
#
# For a model too big for one node, set --nodes-per-vllm K > 1 (one K-node ray cluster per
# endpoint, tensor-parallel over each node's 4 GPUs, pipeline-parallel across the K nodes)
# and bump -N to I*K + J.

#SBATCH --job-name=hpcagent_bench-launch
#SBATCH --nodes=3                 # = inference_endpoints*nodes_per_vllm + judge_nodes
#SBATCH --ntasks-per-node=1       # one launcher rank per node (MPI rank -> role)
#SBATCH --gpus-per-node=4         # a GH200 node
#SBATCH --time=02:00:00
#SBATCH --output=results/launch-%j.out

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
INFERENCE_ENDPOINTS="${INFERENCE_ENDPOINTS:-2}"
NODES_PER_VLLM="${NODES_PER_VLLM:-1}"
JUDGE_NODES="${JUDGE_NODES:-1}"
KERNELS="${KERNELS:-gemm,gesummv}"

# `vllm` is assumed on PATH (a site module / venv); the launcher only places roles.
# srun --mpi=pmix so mpi4py's COMM_WORLD spans the allocation. --ntasks matches the
# allocation so there is exactly one rank per node.
srun --mpi=pmix --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    hpcagent-bench launch openai \
        --model "${MODEL}" \
        --inference-endpoints "${INFERENCE_ENDPOINTS}" \
        --nodes-per-vllm "${NODES_PER_VLLM}" \
        --judge-nodes "${JUDGE_NODES}" \
        --gpus-per-node 4 \
        --kernels "${KERNELS}" \
        --baseline auto --preset S \
        --output "results/agent_launch-${SLURM_JOB_ID}.jsonl"
