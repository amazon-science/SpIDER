#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SpIDER launcher for locally served models (one vLLM server per GPU).
# Drives `python -m spider.run_vllm`.
#
# For the hosted-API equivalent (no GPUs), see run_spider_api.sh -- same
# pipeline, same output schema, different LLM transport.
#
# Usage:
#   ./run_spider.sh                                  # contains@4, GPUs 0-3, all 4 languages
#   GPUS="1 2 3 4 5 6 7" ./run_spider.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./run_spider.sh
#   LANGS="python java" ./run_spider.sh
#   ./run_spider.sh --skip_server_launch             # servers already up
#
# Graph exploration -- each edge type expands independently from the seed at its
# own depth, then the results are unioned (mixed-type paths are NOT traversed).
# Results go to a per-edge-config directory, so these never overwrite each other:
#
#   ./run_spider.sh                                                   # spider_edges-contains4_nbrcap200/
#   EDGES=invokes BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider.sh
#                                                                     # spider_edges-invokes2_nbrcap200/
#   EDGES=contains,invokes BFS_DEPTH_PER_EDGE=invokes:2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider.sh
#                                                                     # spider_edges-contains4+invokes2_nbrcap200/
#
# Edge-type notes (mean function-neighbors per center, SWE-PolyBench/python):
#   contains@4   11     the paper default
#   imports@4     5
#   inherits@4    0     finds NO function-type neighbors -- alone it degrades the
#                       run to the plain-ranking baseline. Only useful in a union.
#   invokes@2    34
#   invokes@4   181     ~70k tokens of code at mc1500 -- overflows MAX_MODEL_LEN=65536
#                       unless capped. ALWAYS pair depth>2 invokes with
#                       MAX_NEIGHBORS_PER_CENTER.
# Overflowing calls are caught and counted as `num_call_errors` in the results,
# but they silently contribute no neighbors -- check that field before trusting a run.
#
# The `bfs4` in the score-pickle filename refers to how the 500-node candidate
# pool was precomputed; it does NOT constrain BFS_DEPTH here, which is re-run
# live against the graph.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_HOME="${SPIDER_HOME:-$(dirname "$SCRIPT_DIR")}"

# --- Data locations (override any of these for your own layout) ---
DATA_ROOT="${DATA_ROOT:-$SPIDER_HOME/data}"
GRAPHS_ROOT="${GRAPHS_ROOT:-$DATA_ROOT/SpIDER-Bench}"
SCORES_ROOT="${SCORES_ROOT:-$DATA_ROOT/node_embeddings}"
RESULTS_ROOT="${RESULTS_ROOT:-$SPIDER_HOME/results}"

cd "$SPIDER_HOME"

GPUS="${GPUS:-0 1 2 3}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_DIR_NAME="$(basename "$MODEL")"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
LANGS="${LANGS:-python java javascript typescript}"

DATASET="${DATASET:-swepolybench}"
case "$DATASET" in
    swepolybench)      ITEMS="$LANGS" ;;
    swebench_verified) ITEMS="${SUBSETS:-test}" ;;
    *) echo "[spider] ERROR: unknown DATASET=$DATASET (use swepolybench|swebench_verified)" >&2; exit 1 ;;
esac

# --- Graph exploration ---
EDGES="${EDGES:-contains}"                 # comma-separated: contains,imports,inherits,invokes
BFS_DEPTH="${BFS_DEPTH:-4}"                # uniform depth for every edge above
BFS_DEPTH_PER_EDGE="${BFS_DEPTH_PER_EDGE:-}"   # overrides, e.g. "invokes:2"
MAX_NEIGHBORS_PER_CENTER="${MAX_NEIGHBORS_PER_CENTER:-200}"   # 0 = unlimited

# --- Hyperparameters ---
K="${K:-20}"; L="${L:-20}"; N="${N:-500}"; C="${C:-5}"
MAX_STEPS="${MAX_STEPS:-1}"
MAX_CODE_CHARS="${MAX_CODE_CHARS:-1500}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"

OPT_ARGS=(--edges "$EDGES" --bfs_depth "$BFS_DEPTH"
          --max_neighbors_per_center "$MAX_NEIGHBORS_PER_CENTER")
[ -n "$BFS_DEPTH_PER_EDGE" ] && OPT_ARGS+=(--bfs_depth_per_edge "$BFS_DEPTH_PER_EDGE")
[ -n "$LIMIT" ] && OPT_ARGS+=(--limit "$LIMIT")
# INSTANCE_IDS is a space-separated list; splitting it is the point.
# shellcheck disable=SC2206
[ -n "${INSTANCE_IDS:-}" ] && OPT_ARGS+=(--instance_ids ${INSTANCE_IDS})

echo "[spider] model=$MODEL  gpus=$GPUS  max_model_len=$MAX_MODEL_LEN"
echo "[spider] opt_args: ${OPT_ARGS[*]}"
echo "[spider] dataset=$DATASET  items=\"$ITEMS\"  K=$K L=$L N=$N C=$C max_steps=$MAX_STEPS"

for ITEM in $ITEMS; do
    case "$DATASET" in
        swepolybench)
            SCORES_PATH="${SCORES_PATH_OVERRIDE:-$SCORES_ROOT/swepolybench/test_${ITEM}_issue_node_scores_bfs4_filtered500.pkl}"
            GRAPHS_DIR="${GRAPHS_DIR_OVERRIDE:-$GRAPHS_ROOT/SWE-PolyBench/${ITEM}}"
            OUT_DIR="$RESULTS_ROOT/${MODEL_DIR_NAME}/swepolybench/${ITEM}/spider" ;;
        swebench_verified)
            SCORES_PATH="${SCORES_PATH_OVERRIDE:-$SCORES_ROOT/swebench_verified/${ITEM}_issue_node_scores_bfs4_filtered500.pkl}"
            GRAPHS_DIR="${GRAPHS_DIR_OVERRIDE:-$GRAPHS_ROOT/SWE-bench_Verified}"
            OUT_DIR="$RESULTS_ROOT/${MODEL_DIR_NAME}/swebench_verified/${ITEM}/spider" ;;
    esac

    if [ ! -f "$SCORES_PATH" ]; then
        echo "[spider] WARN: scores_path missing -- $SCORES_PATH -- skipping $ITEM"
        continue
    fi
    if [ ! -d "$GRAPHS_DIR" ]; then
        echo "[spider] WARN: graphs_dir missing -- $GRAPHS_DIR -- skipping $ITEM"
        continue
    fi

    echo
    echo "================================================================================"
    echo "[spider] $DATASET / $ITEM"
    echo "================================================================================"

    # shellcheck disable=SC2086
    python -m spider.run_vllm \
        --gpus $GPUS \
        --model "$MODEL" \
        --max_model_len $MAX_MODEL_LEN \
        --max_tokens $MAX_TOKENS \
        --gpu_mem_util 0.9 \
        --node_type function \
        --ranking_type dense \
        "${OPT_ARGS[@]}" \
        --K "$K" --L "$L" --N "$N" --C "$C" \
        --max_code_chars "$MAX_CODE_CHARS" \
        --max_steps "$MAX_STEPS" \
        --temperature "$TEMPERATURE" \
        --startup_timeout "$STARTUP_TIMEOUT" \
        --scores_path "$SCORES_PATH" \
        --graphs_dir "$GRAPHS_DIR" \
        --output_dir "$OUT_DIR" \
        "$@"
done

echo
echo "[spider] all done."
