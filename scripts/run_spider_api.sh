#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SpIDER launcher against a hosted LLM endpoint (no GPUs).
#
# Instances are sharded across a thread pool firing concurrent API calls.
# Drives `python -m spider.run_api`.
#
# Output layout (the edge slug is appended to the leaf dir by the Python runner):
#   $RESULTS_ROOT/<model>/<dataset>/<item>/spider_edges-<slug>[_nbrcap<n>]/
#       results_<tag>_ours.json     LocAgent GT
#       results_<tag>_spider.json   modified_nodes n candidate pool
#
# Credentials: never hardcoded. Export LLM_API_KEY (and LLM_BASE_URL for a
# non-OpenAI endpoint) before running:
#   export LLM_API_KEY=...
#
# Usage:
#   ./run_spider_api.sh                                  # contains@4, all 4 languages
#   LANGS="python java" ./run_spider_api.sh
#   LIMIT=5 LANGS=python ./run_spider_api.sh             # smoke test
#   EDGES=invokes BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
#   EDGES=contains,invokes BFS_DEPTH_PER_EDGE=contains:4,invokes:2 \
#       MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
#   DATASET=swebench_verified ./run_spider_api.sh
#
# Cost: C=5 calls per instance per round. All four SWE-PolyBench languages =
#   2089 instances (python 199, java 165, javascript 1017, typescript 708) =
#   ~10.4k calls at MAX_STEPS=1. Scope a trial run with LIMIT / LANGS first.
#
# Edge-type notes (mean function-neighbors per center, SWE-PolyBench/python):
#   contains@4   11     the paper default
#   imports@4     5
#   inherits@4    0     finds NO function-type neighbors -- alone it degrades the
#                       run to the plain-ranking baseline. Only useful in a union.
#   invokes@2    34
#   invokes@4   181     ~70k tokens of code at mc1500. ALWAYS pair depth>2
#                       invokes with MAX_NEIGHBORS_PER_CENTER.
# After any run, check `num_call_errors` / `num_empty_responses` in the results --
# both mean Step B silently contributed nothing for those centers, leaving the
# plain-ranking baseline behind.
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

# --- LLM endpoint ---
LLM_PROVIDER="${LLM_PROVIDER:-azure_responses}"   # azure_responses | azure | openai | vllm
LLM_MODEL="${LLM_MODEL:?set LLM_MODEL to your model / deployment name}"
LLM_BASE_URL="${LLM_BASE_URL:-}"
if [ -z "${LLM_API_KEY:-}" ]; then
    echo "[spider-api] ERROR: LLM_API_KEY is not set. Export it before running:" >&2
    echo "               export LLM_API_KEY=..." >&2
    exit 1
fi

MODEL_DIR_NAME="$(basename "$LLM_MODEL")"
LANGS="${LANGS:-python java javascript typescript}"

# DATASET selects the benchmark and the items iterated over:
#   swepolybench      -> items = $LANGS (python java javascript typescript)
#   swebench_verified -> items = $SUBSETS (default "test"); single python set
DATASET="${DATASET:-swepolybench}"
case "$DATASET" in
    swepolybench)      ITEMS="$LANGS" ;;
    swebench_verified) ITEMS="${SUBSETS:-test}" ;;
    *) echo "[spider-api] ERROR: unknown DATASET=$DATASET (use swepolybench|swebench_verified)" >&2; exit 1 ;;
esac

MAX_TOKENS="${MAX_TOKENS:-8192}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# --- Graph exploration ---
EDGES="${EDGES:-contains}"                 # comma-separated: contains,imports,inherits,invokes
BFS_DEPTH="${BFS_DEPTH:-4}"                # uniform depth for every edge above
BFS_DEPTH_PER_EDGE="${BFS_DEPTH_PER_EDGE:-}"   # overrides, e.g. "invokes:2"
MAX_NEIGHBORS_PER_CENTER="${MAX_NEIGHBORS_PER_CENTER:-200}"   # 0 = unlimited

# --- Hyperparameters ---
K="${K:-20}"; L="${L:-20}"; N="${N:-500}"; C="${C:-5}"
MAX_STEPS="${MAX_STEPS:-1}"
MAX_CODE_CHARS="${MAX_CODE_CHARS:-1500}"
LIMIT="${LIMIT:-}"

cd "$SPIDER_HOME"

echo "[spider-api] llm_provider=$LLM_PROVIDER  llm_model=$LLM_MODEL"
echo "[spider-api] llm_base_url=${LLM_BASE_URL:-<provider default>}"
echo "[spider-api] edges=$EDGES  bfs_depth=$BFS_DEPTH  bfs_depth_per_edge=${BFS_DEPTH_PER_EDGE:-<none>}"
echo "[spider-api] max_neighbors_per_center=$MAX_NEIGHBORS_PER_CENTER  max_tokens=$MAX_TOKENS  workers=$NUM_WORKERS"
echo "[spider-api] dataset=$DATASET  items=\"$ITEMS\"  K=$K L=$L N=$N C=$C max_steps=$MAX_STEPS  limit=${LIMIT:-<all>}"

# Optional flags, collected in an array so a URL with '?'/'&' survives intact.
OPT_ARGS=()
[ -n "$BFS_DEPTH_PER_EDGE" ] && OPT_ARGS+=(--bfs_depth_per_edge "$BFS_DEPTH_PER_EDGE")
[ -n "$LIMIT" ] && OPT_ARGS+=(--limit "$LIMIT")
[ -n "$LLM_BASE_URL" ] && OPT_ARGS+=(--api_base_url "$LLM_BASE_URL")
# INSTANCE_IDS is a space-separated list; splitting it is the point.
# shellcheck disable=SC2206
[ -n "${INSTANCE_IDS:-}" ] && OPT_ARGS+=(--instance_ids ${INSTANCE_IDS})

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
        echo "[spider-api] WARN: scores_path missing -- $SCORES_PATH -- skipping $ITEM"
        continue
    fi
    if [ ! -d "$GRAPHS_DIR" ]; then
        echo "[spider-api] WARN: graphs_dir missing -- $GRAPHS_DIR -- skipping $ITEM"
        continue
    fi

    echo
    echo "================================================================================"
    echo "[spider-api] $DATASET / $ITEM"
    echo "================================================================================"

    python -m spider.run_api \
        --provider "$LLM_PROVIDER" \
        --model    "$LLM_MODEL" \
        --api_key  "$LLM_API_KEY" \
        --temperature "$TEMPERATURE" \
        --max_tokens  "$MAX_TOKENS" \
        --node_type function \
        --ranking_type dense \
        --edges "$EDGES" \
        --bfs_depth "$BFS_DEPTH" \
        --max_neighbors_per_center "$MAX_NEIGHBORS_PER_CENTER" \
        "${OPT_ARGS[@]}" \
        --K "$K" --L "$L" --N "$N" --C "$C" \
        --max_code_chars "$MAX_CODE_CHARS" \
        --max_steps "$MAX_STEPS" \
        --num_workers "$NUM_WORKERS" \
        --scores_path "$SCORES_PATH" \
        --graphs_dir  "$GRAPHS_DIR" \
        --output_dir  "$OUT_DIR" \
        "$@"
done

echo
echo "[spider-api] all done."
