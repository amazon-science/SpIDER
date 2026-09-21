#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Every SpIDER run behind the reported tables, in the order they were produced.
#
# Requires LLM_API_KEY (and LLM_MODEL / LLM_BASE_URL) in the environment -- see
# run_spider_api.sh. Each line writes to its own edge-slug directory, so lines
# are independent and safe to re-run or comment out individually.
#
# Cost warning: the full sweep is ~33 runs over 2589 instances at C=5 calls per
# instance -- on the order of 10^5 API calls. Run one line at a time.
#
# After the runs, build the tables:
#   python summarize.py --root results/<model>
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Main table: SWE-PolyBench, all four languages, neighbor cap 200.
# ---------------------------------------------------------------------------
EDGES=contains BFS_DEPTH=4 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
EDGES=invokes  BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
EDGES=imports  BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
EDGES=contains,invokes BFS_DEPTH_PER_EDGE=contains:4,invokes:2 \
    MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh

# `inherits` yields zero function-type neighbors, so it reproduces the plain
# dense-retrieval baseline. Reported for python only.
LANGS=python EDGES=inherits BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
LANGS=python EDGES=inherits BFS_DEPTH=4 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh

# ---------------------------------------------------------------------------
# Ablations: BFS depth and neighbor cap on the `invokes` edge.
# ---------------------------------------------------------------------------
EDGES=invokes BFS_DEPTH=1 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
EDGES=invokes BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=100 ./run_spider_api.sh
EDGES=invokes BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=50  ./run_spider_api.sh

# ---------------------------------------------------------------------------
# SWE-bench Verified (single python set; LocAgent GT view only -- these
# instances carry no `modified_nodes` field, so no _spider.json is written).
# ---------------------------------------------------------------------------
DATASET=swebench_verified EDGES=contains BFS_DEPTH=4 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
DATASET=swebench_verified EDGES=invokes  BFS_DEPTH=2 MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh
DATASET=swebench_verified EDGES=contains,invokes BFS_DEPTH_PER_EDGE=contains:4,invokes:2 \
    MAX_NEIGHBORS_PER_CENTER=200 ./run_spider_api.sh

echo "[reproduce] all sweeps done."
