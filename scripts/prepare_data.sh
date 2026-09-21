#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the score pickle SpIDER runs on, for one dataset item.
#
#   stage 1  embed_graph_nodes          graphs      -> *.node_embeddings.pkl
#   stage 2  compute_issue_node_scores  embeddings  -> *_issue_node_scores.pkl
#   stage 3  add_filtered_bfs_rankings  scores      -> *_issue_node_scores_bfs4_filtered500.pkl
#
# Stage 1 is skippable: SpIDER-Bench ships precomputed *.npz node embeddings, and
# stage 2 reads them directly with EMB_FORMAT=npz USE_PRECOMPUTED_SCORES=1 --
# that is how the released numbers were produced, and it needs no GPU.
#
# Usage:
#   LANG=python ./prepare_data.sh                      # full chain, own embeddings
#   LANG=python EMB_FORMAT=npz USE_PRECOMPUTED_SCORES=1 \
#       EMBEDDINGS_DIR=.../inst_node_embs_... ./prepare_data.sh   # skip stage 1
#   LANG=python STAGES="3" ./prepare_data.sh           # re-run one stage
#   DATASET=swebench_verified ./prepare_data.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_HOME="${SPIDER_HOME:-$(dirname "$SCRIPT_DIR")}"
cd "$SPIDER_HOME"

DATA_ROOT="${DATA_ROOT:-$SPIDER_HOME/data}"
GRAPHS_ROOT="${GRAPHS_ROOT:-$DATA_ROOT/SpIDER-Bench}"
SCORES_ROOT="${SCORES_ROOT:-$DATA_ROOT/node_embeddings}"

DATASET="${DATASET:-swepolybench}"
LANG="${LANG:-python}"
STAGES="${STAGES:-1 2 3}"

MODEL_NAME="${MODEL_NAME:-Salesforce/SweRankEmbed-Small}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DEVICE="${DEVICE:-auto}"

EMB_FORMAT="${EMB_FORMAT:-auto}"                    # auto | pkl | npz
USE_PRECOMPUTED_SCORES="${USE_PRECOMPUTED_SCORES:-0}"

BFS_DEPTH="${BFS_DEPTH:-4}"
FILTER_TOP_N="${FILTER_TOP_N:-500}"
K_VALUES="${K_VALUES:-1,3,5,10,20,50,100,500}"
MAX_WORKERS="${MAX_WORKERS:-64}"

case "$DATASET" in
    swepolybench)
        HF_DATASET="${HF_DATASET:-AmazonScience/SWE-PolyBench}"
        SPLIT="${SPLIT:-test}"
        GRAPHS_DIR="${GRAPHS_DIR:-$GRAPHS_ROOT/SWE-PolyBench/${LANG}}"
        EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-$SCORES_ROOT/swepolybench/${LANG}}"
        SCORES_BASE="$SCORES_ROOT/swepolybench/test_${LANG}_issue_node_scores" ;;
    swebench_verified)
        HF_DATASET="${HF_DATASET:-princeton-nlp/SWE-bench_Verified}"
        SPLIT="${SPLIT:-test}"
        GRAPHS_DIR="${GRAPHS_DIR:-$GRAPHS_ROOT/SWE-bench_Verified}"
        EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-$SCORES_ROOT/swebench_verified/${SPLIT}}"
        SCORES_BASE="$SCORES_ROOT/swebench_verified/${SPLIT}_issue_node_scores" ;;
    *) echo "[prepare] ERROR: unknown DATASET=$DATASET" >&2; exit 1 ;;
esac

SCORES_PATH="${SCORES_BASE}.pkl"
FILTERED_PATH="${SCORES_BASE}_bfs${BFS_DEPTH}_filtered${FILTER_TOP_N}.pkl"

echo "[prepare] dataset=$DATASET item=${LANG}  stages=\"$STAGES\""
echo "[prepare] graphs_dir=$GRAPHS_DIR"
echo "[prepare] embeddings_dir=$EMBEDDINGS_DIR  format=$EMB_FORMAT  precomputed_scores=$USE_PRECOMPUTED_SCORES"
echo "[prepare] scores=$SCORES_PATH"
echo "[prepare] filtered=$FILTERED_PATH"

if [[ " $STAGES " == *" 1 "* ]]; then
    echo
    echo "--- stage 1: embedding graph nodes -------------------------------------"
    mkdir -p "$EMBEDDINGS_DIR"
    python -m data_prep.embed_graph_nodes \
        --graph_dir "$GRAPHS_DIR" \
        --output_dir "$EMBEDDINGS_DIR" \
        --model_name "$MODEL_NAME" \
        --backend hf \
        --batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --max_length "$MAX_LENGTH" \
        --device "$DEVICE" \
        --normalize \
        --trust_remote_code
fi

if [[ " $STAGES " == *" 2 "* ]]; then
    echo
    echo "--- stage 2: issue-node scores -----------------------------------------"
    EXTRA=""
    [ "$EMB_FORMAT" != "auto" ] && EXTRA="$EXTRA --embeddings_format $EMB_FORMAT"
    if [ "$USE_PRECOMPUTED_SCORES" = "1" ]; then
        EXTRA="$EXTRA --use_precomputed_scores"
    else
        EXTRA="$EXTRA --model_name $MODEL_NAME --backend hf --device $DEVICE --max_length $MAX_LENGTH --trust_remote_code"
    fi
    # shellcheck disable=SC2086
    python -m data_prep.compute_issue_node_scores \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --graphs_dir "$GRAPHS_DIR" \
        --dataset "$HF_DATASET" \
        --split "$SPLIT" \
        --output_path "$SCORES_PATH" \
        $EXTRA
fi

if [[ " $STAGES " == *" 3 "* ]]; then
    echo
    echo "--- stage 3: BFS-filtered candidate pool -------------------------------"
    python -m data_prep.add_filtered_bfs_rankings \
        --scores_path "$SCORES_PATH" \
        --output_path "$FILTERED_PATH" \
        --graphs_dir "$GRAPHS_DIR" \
        --k_values "$K_VALUES" \
        --depth "$BFS_DEPTH" \
        --filter_top_n "$FILTER_TOP_N" \
        --max_workers "$MAX_WORKERS"
fi

echo
echo "[prepare] done -> $FILTERED_PATH"
