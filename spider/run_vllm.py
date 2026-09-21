# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU runner for the SpIDER pipeline against locally served models.

Launches one vLLM server per GPU and shards instances round-robin across them.
Same pipeline, same output schema as run_api.py -- only the LLM transport differs.

Prefer the scripts/run_spider.sh launcher; this is the single-dataset entry point.

Usage:
    python -m spider.run_vllm --gpus 0 1 2 3 \
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --edges contains --bfs_depth 4 --max_neighbors_per_center 200 \
        --scores_path .../test_python_issue_node_scores_bfs4_filtered500.pkl \
        --graphs_dir  .../SpIDER-Bench/SWE-PolyBench/python \
        --output_dir  .../results/<model>/swepolybench/python/spider
"""

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from . import summary
from .config import SelectorConfig, parse_edge_depths
from .data_loader import load_scores, load_instance
from .llm_client import ChatClient
from .multigpu_utils import launch_all_servers, stop_servers, shard_instances, BASE_PORT
from .pipeline import run_spider_pipeline
from .run_api import _failed_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_shard(shard_iids, scores_data, cfg, port, model, gpu_id):
    """Process a shard of instances against one GPU's vLLM server."""
    client = ChatClient(
        client=OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY"),
        model=model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    results = []
    for i, iid in enumerate(shard_iids):
        logger.info(f"[GPU {gpu_id}] ({i+1}/{len(shard_iids)}) {iid}")
        t0 = time.time()
        try:
            instance = load_instance(iid, scores_data, cfg)
            result = run_spider_pipeline(instance, cfg, client)
            result["elapsed_sec"] = time.time() - t0
            status = "FAILED" if result.get("failed") else "ok"
            results.append(result)
            logger.info(f"[GPU {gpu_id}] {iid}: {status} recall={result['recall']:.3f} "
                        f"rounds={result.get('num_rounds', '?')} "
                        f"stop={result.get('stop_reason', '?')} "
                        f"time={result['elapsed_sec']:.1f}s")
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] {iid} THREAD FAILED: {type(e).__name__}: {e}\n"
                         f"{traceback.format_exc()}")
            results.append(_failed_result(iid, e, time.time() - t0))
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU SpIDER runner (local vLLM)")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--max_model_len", type=int, default=65536)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--base_port", type=int, default=BASE_PORT)
    parser.add_argument("--skip_server_launch", action="store_true",
                        help="assume vLLM servers are already listening on the ports")
    parser.add_argument("--startup_timeout", type=int, default=500,
                        help="seconds to wait per GPU for vLLM readiness")

    # Data
    parser.add_argument("--scores_path", type=str, required=True,
                        help="issue-node score pickle (see data_prep/)")
    parser.add_argument("--graphs_dir", type=str, required=True,
                        help="directory of per-instance repository graphs, {instance_id}.pkl")

    # Pipeline hyperparameters
    parser.add_argument("--node_type", type=str, default="function",
                        choices=["file", "class", "function"])
    parser.add_argument("--ranking_type", type=str, default="dense", choices=["dense", "bm25"])
    parser.add_argument("--edge_type", type=str, default="contains",
                        choices=["contains", "imports", "inherits", "invokes"])
    parser.add_argument("--edges", type=str, default=None,
                        help="Multi-edge graph exploration, e.g. 'contains,invokes'. "
                             "Supersedes --edge_type when set. Each edge expands "
                             "independently from the seed at its own depth.")
    parser.add_argument("--bfs_depth", type=int, default=4,
                        help="Uniform BFS depth for every explored edge type.")
    parser.add_argument(
        "--bfs_depth_per_edge", type=str, default="",
        help="Per-edge-type BFS depth overrides. Format: 'contains:4,invokes:2'. "
             "Edge types not listed fall back to --bfs_depth. CRUCIAL when adding "
             "dense edges like `invokes`: depth 4 over invokes can explode the BFS "
             "frontier by ~100x. Try `invokes:2`.")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--C", type=int, default=5)
    parser.add_argument("--max_code_chars", type=int, default=1500)
    parser.add_argument("--max_neighbors_per_center", type=int, default=0,
                        help="Cap BFS neighbors shown per center (0 = unlimited). "
                             "Neighbors are rank-ordered, so this drops the "
                             "lowest-ranked. Required with `invokes`, whose BFS can "
                             "cover the whole N-window and overflow --max_model_len.")
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)

    parser.add_argument("--instance_ids", type=str, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N instances (smoke tests)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_edge_slug_dir", action="store_true",
                        help="Write straight into --output_dir instead of appending the "
                             "edge slug to its leaf (e.g. 'spider_edges-contains4').")
    args = parser.parse_args()

    gpus = args.gpus
    num_gpus = len(gpus)

    try:
        bfs_depth_per_edge = parse_edge_depths(args.bfs_depth_per_edge)
    except ValueError as e:
        parser.error(f"--bfs_depth_per_edge: {e}")

    cfg = SelectorConfig(
        scores_path=args.scores_path, graphs_dir=args.graphs_dir,
        node_type=args.node_type, ranking_type=args.ranking_type, edge_type=args.edge_type,
        edges=args.edges, bfs_depth=args.bfs_depth, bfs_depth_per_edge=bfs_depth_per_edge,
        K=args.K, L=args.L, N=args.N, C=args.C,
        max_code_chars=args.max_code_chars, max_steps=args.max_steps,
        max_neighbors_per_center=args.max_neighbors_per_center,
        provider="vllm", model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )

    logger.info(f"Loading scores from {cfg.scores_path}")
    scores_data = load_scores(cfg)
    all_ids = list(scores_data["issues"].keys())
    instance_ids = [iid for iid in (args.instance_ids or all_ids) if iid in scores_data["issues"]]
    if args.limit:
        instance_ids = instance_ids[: args.limit]
    if not instance_ids:
        raise SystemExit("no instances selected -- check --instance_ids / --scores_path")
    logger.info(f"Pipeline: spider  |  {len(instance_ids)} instances  |  {num_gpus} GPUs: {gpus}")
    logger.info(f"Graph exploration: edge_depth_map={cfg.edge_depth_map()} "
                f"max_neighbors_per_center={cfg.max_neighbors_per_center or 'unlimited'}")

    try:
        procs, ports = launch_all_servers(
            gpus, args.model, args.max_model_len, args.gpu_mem_util,
            base_port=args.base_port, skip_launch=args.skip_server_launch,
            startup_timeout=args.startup_timeout,
        )
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    shards = shard_instances(instance_ids, num_gpus)
    for gpu_id, port, shard in zip(gpus, ports, shards):
        logger.info(f"  GPU {gpu_id} (port {port}): {len(shard)} instances")

    model_slug = args.model.replace("/", "_")
    out_dir = summary.resolve_out_dir(cfg, no_edge_slug_dir=args.no_edge_slug_dir)
    tag = (f"spider_{cfg.node_type}_{cfg.ranking_type}_"
           f"{cfg.run_slug()}_K{cfg.K}_L{cfg.L}_"
           f"N{cfg.N}_C{cfg.C}_maxsteps{cfg.max_steps}_mc{cfg.max_code_chars}_{model_slug}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")
    partial_file = out_dir / f"results_{tag}.partial.jsonl"

    all_results = []
    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            futures = {
                pool.submit(process_shard, shard, scores_data, cfg, port, args.model, gpu_id): gpu_id
                for shard, port, gpu_id in zip(shards, ports, gpus)
            }
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    shard_results = future.result()
                    all_results.extend(shard_results)
                    with open(partial_file, "a") as pf:
                        for r in shard_results:
                            pf.write(json.dumps(r, default=str) + "\n")
                    logger.info(f"GPU {gpu_id}: completed {len(shard_results)} instances")
                except Exception as e:
                    logger.error(f"GPU {gpu_id}: shard failed: {e}")
    finally:
        if procs:
            logger.info("Stopping vLLM servers...")
            stop_servers(procs)

    id_order = {iid: i for i, iid in enumerate(instance_ids)}
    all_results.sort(key=lambda r: id_order.get(r["instance_id"], 0))

    views = summary.build_views(all_results)
    n = len(views["results"])
    n_failed = len(views["failed_results"])

    if n:
        summary.log_agg("ours / LocAgent gt", views["ours_agg"], views["ours_n_eval"],
                        "spider", n, n_failed, extra=f", {num_gpus} GPUs")
        if views["has_spider"]:
            summary.log_agg("spider / modnodes n pool", views["spider_agg"],
                            views["spider_n_eval"], "spider", n, n_failed,
                            extra=f", {num_gpus} GPUs")

    tot_err = sum(r.get("num_call_errors", 0) for r in views["results"])
    tot_empty = sum(r.get("num_empty_responses", 0) for r in views["results"])
    if tot_err or tot_empty:
        logger.warning(f"\n{'!'*80}\n  {tot_err} Step-B call errors, {tot_empty} empty responses.\n"
                       f"  Those centers selected no neighbors, so the run partially fell back\n"
                       f"  to the plain ranking baseline. Check --max_model_len / --max_tokens.\n"
                       f"{'!'*80}")

    written = summary.dump_results(out_dir, tag, cfg, "spider", all_results, gpus=gpus)
    logger.info("Results saved to:\n  " + "\n  ".join(str(p) for p in written) +
                f"  ({n} succeeded, {n_failed} failed)")
    if partial_file.exists():
        partial_file.unlink()


if __name__ == "__main__":
    main()
