# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel runner for the SpIDER pipeline against a hosted LLM endpoint.

No GPUs needed: instances are sharded across a thread pool that fires concurrent
API calls, and one shared client is reused (httpx.Client / the OpenAI client are
both thread-safe).

Two ground-truth views are emitted side by side, with a byte-identical `results`
array:
    results_<tag>_ours.json    LocAgent GT (patch-line overlap with node ranges)
    results_<tag>_spider.json  SpIDER GT (`modified_nodes` n candidate pool)
The `_spider.json` file is written only when at least one instance carries a
`modified_nodes` field (SWE-PolyBench does; SWE-bench Verified does not).

The leaf of --output_dir gets the edge-exploration slug appended, so runs over
different edge types never collide -- see summary.resolve_out_dir.

Prefer the scripts/run_spider_api.sh launcher; this is the single-dataset entry
point.

Usage:
    python -m spider.run_api \
        --provider azure_responses --model <deployment> \
        --api_base_url "https://<resource>.openai.azure.com/openai/v1" \
        --api_key "$LLM_API_KEY" \
        --edges contains,invokes --bfs_depth 4 --bfs_depth_per_edge invokes:2 \
        --max_neighbors_per_center 200 \
        --scores_path .../test_python_issue_node_scores_bfs4_filtered500.pkl \
        --graphs_dir  .../SpIDER-Bench/SWE-PolyBench/python \
        --output_dir  .../results/<model>/swepolybench/python/spider
    # -> .../python/spider_edges-contains4+invokes2_nbrcap200/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from . import summary
from .config import SelectorConfig, parse_edge_depths
from .data_loader import load_scores, load_instance
from .llm_client import build_client
from .pipeline import run_spider_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _failed_result(iid: str, exc: Exception, elapsed: float) -> Dict:
    """Failure stub matching the succeeded-result schema, so aggregation and the
    summarizer can treat failures as ordinary rows and count them separately."""
    return {
        "instance_id": iid, "failed": True,
        "error": {"type": type(exc).__name__, "message": str(exc),
                  "round_attempted": -1, "traceback": traceback.format_exc()},
        "selected": [], "num_selected": 0,
        "ground_truth": [], "hits": [],
        "recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "accuracy": 0.0,
        "first_round_recall": 0.0, "first_round_hit_rate": 0.0,
        "first_round_accuracy": 0.0,
        "spider_ground_truth": [], "spider_hits": [], "spider_evaluated": False,
        "spider_recall": 0.0, "spider_precision": 0.0,
        "spider_hit_rate": 0.0, "spider_accuracy": 0.0,
        "spider_first_round_recall": 0.0, "spider_first_round_hit_rate": 0.0,
        "num_rounds": 0, "early_stop": False, "stop_reason": "thread_error",
        "confident": False, "total_nbr_selected": 0, "total_nbr_truncated": 0,
        "num_call_errors": 0, "num_empty_responses": 0, "total_seen": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0, "num_calls": 0},
        "rounds": [], "elapsed_sec": elapsed,
    }


def _process_instance(iid, scores_data, cfg, client, partial_fh, write_lock) -> Dict:
    t0 = time.time()
    try:
        instance = load_instance(iid, scores_data, cfg)
        result = run_spider_pipeline(instance, cfg, client)
        result["elapsed_sec"] = time.time() - t0
    except Exception as e:
        logger.error(f"{iid} THREAD FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        result = _failed_result(iid, e, time.time() - t0)

    # Threads share one handle; serialize the append so lines never interleave.
    with write_lock:
        partial_fh.write(json.dumps(result, default=str) + "\n")
        partial_fh.flush()
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Parallel runner for the SpIDER pipeline against a hosted LLM API")

    # --- LLM endpoint ---
    ap.add_argument("--provider", type=str, default="azure_responses",
                    choices=["openai", "azure_responses", "azure", "vllm"])
    ap.add_argument("--model", type=str, required=True,
                    help="model / deployment name")
    ap.add_argument("--api_base_url", type=str, default=None)
    ap.add_argument("--api_key", type=str, default=os.environ.get("LLM_API_KEY"),
                    help="defaults to $LLM_API_KEY; never hardcode a key here")
    ap.add_argument("--azure_api_base", type=str, default=None)
    ap.add_argument("--azure_api_version", type=str, default=None)
    ap.add_argument("--azure_api_key", type=str, default=os.environ.get("AZURE_API_KEY"))
    ap.add_argument("--azure_deployment", type=str, default=None)
    ap.add_argument("--vllm_base_url", type=str, default=None)
    ap.add_argument("--vllm_api_key", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=8192,
                    help="max output tokens. Reasoning models spend this budget on "
                         "reasoning too, and a starved call returns no text at all -- "
                         "which parses as an empty selection. If you see empty=N>0 in "
                         "the logs, raise it.")

    # --- Data ---
    ap.add_argument("--scores_path", type=str, required=True,
                    help="issue-node score pickle (see data_prep/)")
    ap.add_argument("--graphs_dir", type=str, required=True,
                    help="directory of per-instance repository graphs, {instance_id}.pkl")

    # --- Pipeline hyperparameters (defaults reproduce the paper) ---
    ap.add_argument("--node_type", type=str, default="function",
                    choices=["file", "class", "function"])
    ap.add_argument("--ranking_type", type=str, default="dense", choices=["dense", "bm25"])
    ap.add_argument("--edge_type", type=str, default="contains",
                    choices=["contains", "imports", "inherits", "invokes"])
    ap.add_argument("--edges", type=str, default=None,
                    help="Multi-edge exploration, e.g. 'contains,invokes'. "
                         "Supersedes --edge_type when set.")
    ap.add_argument("--bfs_depth", type=int, default=4)
    ap.add_argument("--bfs_depth_per_edge", type=str, default="",
                    help="Per-edge depth overrides, e.g. 'contains:4,invokes:2'.")
    ap.add_argument("--max_neighbors_per_center", type=int, default=0,
                    help="Cap BFS neighbors shown per center (0 = unlimited). "
                         "Required for dense edges like `invokes`.")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--C", type=int, default=5)
    ap.add_argument("--max_code_chars", type=int, default=1500)
    ap.add_argument("--max_steps", type=int, default=1)

    # --- Parallelism / output ---
    ap.add_argument("--num_workers", type=int, default=8,
                    help="threads firing concurrent API calls")
    ap.add_argument("--instance_ids", type=str, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N instances (smoke tests)")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--no_edge_slug_dir", action="store_true",
                    help="Write straight into --output_dir instead of appending the "
                         "edge slug to its leaf (e.g. 'spider_edges-contains4').")
    ap.add_argument("--partial_dir", type=str, default=None,
                    help="dir for the streaming partial JSONL (default: the resolved output dir)")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    try:
        bfs_depth_per_edge = parse_edge_depths(args.bfs_depth_per_edge)
    except ValueError as e:
        ap.error(f"--bfs_depth_per_edge: {e}")

    if args.provider in ("azure_responses", "openai") and not args.api_key:
        ap.error(f"--provider {args.provider} needs a key: pass --api_key or set $LLM_API_KEY")

    cfg = SelectorConfig(
        scores_path=args.scores_path, graphs_dir=args.graphs_dir,
        node_type=args.node_type, ranking_type=args.ranking_type,
        edge_type=args.edge_type, edges=args.edges,
        bfs_depth=args.bfs_depth, bfs_depth_per_edge=bfs_depth_per_edge,
        K=args.K, L=args.L, N=args.N, C=args.C,
        max_code_chars=args.max_code_chars, max_steps=args.max_steps,
        max_neighbors_per_center=args.max_neighbors_per_center,
        provider=args.provider, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        api_base_url=args.api_base_url, api_key=args.api_key,
        azure_api_base=args.azure_api_base, azure_api_version=args.azure_api_version,
        azure_api_key=args.azure_api_key, azure_deployment=args.azure_deployment,
        vllm_base_url=args.vllm_base_url, vllm_api_key=args.vllm_api_key,
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

    client = build_client(cfg)
    logger.info(f"Pipeline: spider  |  {len(instance_ids)} instances  |  "
                f"{args.num_workers} workers")
    logger.info(f"LLM: client={type(client).__name__} provider={args.provider} "
                f"model={args.model} temperature={args.temperature} max_tokens={args.max_tokens}")
    logger.info(f"Graph exploration: edge_depth_map={cfg.edge_depth_map()} "
                f"max_neighbors_per_center={cfg.max_neighbors_per_center or 'unlimited'}")

    out_dir = summary.resolve_out_dir(cfg, no_edge_slug_dir=args.no_edge_slug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")
    model_slug = args.model.replace("/", "_")
    tag = (f"spider_{cfg.node_type}_{cfg.ranking_type}_{cfg.run_slug()}_"
           f"K{cfg.K}_L{cfg.L}_N{cfg.N}_C{cfg.C}_maxsteps{cfg.max_steps}_"
           f"mc{cfg.max_code_chars}_api-{model_slug}")
    if args.tag:
        tag += f"_{args.tag}"

    partial_dir = Path(args.partial_dir) if args.partial_dir else out_dir
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_file = partial_dir / f"results_{tag}.partial.jsonl"
    if partial_file.exists():
        partial_file.unlink()
    logger.info(f"partial -> {partial_file}")

    write_lock = threading.Lock()
    all_results: List[Dict] = []
    t_start = time.time()
    with open(partial_file, "a") as pf, ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(_process_instance, iid, scores_data, cfg, client, pf, write_lock): iid
            for iid in instance_ids
        }
        for i, fut in enumerate(as_completed(futures)):
            iid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # _process_instance already guards; belt and braces
                logger.error(f"{iid} unexpected: {type(e).__name__}: {e}")
                continue
            all_results.append(r)
            status = "FAILED" if r.get("failed") else "ok"
            warn = ""
            if r.get("num_call_errors"):
                warn += f" call_errors={r['num_call_errors']}"
            if r.get("num_empty_responses"):
                warn += f" empty={r['num_empty_responses']}"
            u = r.get("usage", {})
            logger.info(f"({i+1}/{len(instance_ids)}) {iid}: {status} "
                        f"recall={r['recall']:.3f} spider_recall={r.get('spider_recall', 0.0):.3f} "
                        f"rounds={r.get('num_rounds','?')} time={r['elapsed_sec']:.1f}s "
                        f"tok(in/out/reason)={u.get('prompt_tokens',0)}/{u.get('completion_tokens',0)}/"
                        f"{u.get('reasoning_tokens',0)}{warn}")

    # Deterministic order regardless of completion order.
    id_order = {iid: i for i, iid in enumerate(instance_ids)}
    all_results.sort(key=lambda r: id_order.get(r["instance_id"], 0))

    views = summary.build_views(all_results)
    n, n_failed = len(views["results"]), len(views["failed_results"])
    if n:
        summary.log_agg("ours / LocAgent gt", views["ours_agg"], views["ours_n_eval"],
                        "spider", n, n_failed)
        if views["has_spider"]:
            summary.log_agg("spider / modnodes n pool", views["spider_agg"],
                            views["spider_n_eval"], "spider", n, n_failed)

    # Degradation guards: these mean Step B silently contributed nothing.
    tot_err = sum(r.get("num_call_errors", 0) for r in views["results"])
    tot_empty = sum(r.get("num_empty_responses", 0) for r in views["results"])
    if tot_err or tot_empty:
        logger.warning(f"\n{'!'*80}\n  {tot_err} Step-B call errors, {tot_empty} empty responses.\n"
                       f"  Those centers selected no neighbors, so the run partially fell back\n"
                       f"  to the plain ranking baseline. Check --max_tokens / context length.\n"
                       f"{'!'*80}")

    written = summary.dump_results(out_dir, tag, cfg, "spider", all_results, gpus=[])
    logger.info("Results saved to:\n  " + "\n  ".join(str(p) for p in written) +
                f"  ({n} succeeded, {n_failed} failed, {time.time()-t_start:.0f}s wall)")
    if partial_file.exists():
        partial_file.unlink()


if __name__ == "__main__":
    main()
