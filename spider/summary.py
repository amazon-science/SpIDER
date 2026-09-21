# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregation + result serialization shared by both SpIDER runners.

The hosted-API runner (run_api.py) and the local vLLM runner (run_vllm.py) go
through this module, so they emit a byte-identical schema.

Two GT views:
  - "ours"   : LocAgent GT (patch-line overlap with graph node ranges)
  - "spider" : SWE-PolyBench `modified_nodes` intersected with the candidate pool.
               Only emitted when the results carry the spider_* fields.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Config fields holding live credentials. `vars(cfg)` is written into every
# results JSON, so these are masked before serialization.
_SECRET_FIELDS = ("api_key", "azure_api_key", "vllm_api_key")


def resolve_out_dir(cfg, no_edge_slug_dir: bool = False) -> Path:
    """Append the graph-exploration slug to the leaf output dir.

    `.../python/spider` -> `.../python/spider_edges-contains4+invokes2_nbrcap40`,
    so runs over different edge types/depths land in separate directories rather
    than sharing one and being told apart only by their filenames.

    Mutates `cfg.output_dir` so the config block serialized into the results JSON
    names the directory actually written.
    """
    out_dir = Path(cfg.output_dir)
    if not no_edge_slug_dir:
        out_dir = out_dir.with_name(f"{out_dir.name}_{cfg.run_slug()}")
    cfg.output_dir = str(out_dir)
    return out_dir


def redact_config(cfg) -> Dict:
    """`vars(cfg)` with credential fields masked."""
    d = dict(vars(cfg))
    for k in _SECRET_FIELDS:
        if d.get(k):
            d[k] = "***REDACTED***"
    return d


def build_agg(results: List[Dict], failed_results: List[Dict], all_results: List[Dict],
              view_name: str, eval_filter, recall_f: str, precision_f: str,
              hit_rate_f: str, accuracy_f: str,
              first_round_recall_f: str = "first_round_recall",
              first_round_hit_rate_f: str = "first_round_hit_rate",
              baseline_prefix: str = "baseline_") -> Tuple[Dict, int]:
    """Aggregate over succeeded instances.

    Eval metrics (recall/precision/hit_rate/accuracy) divide by # of evaluated
    instances -- i.e. succeeded AND with a non-empty granularity-specific GT --
    matching evaluate_localization_results.py. Process stats (avg_selected/rounds/
    time/tokens/stop reasons) keep dividing by `n` (all succeeded) since they
    describe pipeline behavior, not retrieval.

    `baseline_prefix` names the no-LLM retrieval control fields (top-L of the
    ranking) emitted per instance by the spider pipeline. Absent for pipelines
    that don't emit them, in which case the baseline block is omitted rather
    than reported as zeros -- a 0.0 baseline would read as a real measurement.
    """
    n = len(results)
    evaluated_ = [r for r in results if eval_filter(r)]
    n_eval_ = len(evaluated_)
    if not n:
        return {}, n_eval_
    a = {
        "gt_source": view_name,
        "num_instances": n,
        "num_issues_evaluated": n_eval_,
        "num_no_gt": n - n_eval_,
        "num_failed": len(failed_results),
        "num_total_attempted": len(all_results),
        "avg_recall":    (sum(r[recall_f] for r in evaluated_) / n_eval_) if n_eval_ else 0.0,
        "avg_precision": (sum(r[precision_f] for r in evaluated_) / n_eval_) if n_eval_ else 0.0,
        "avg_hit_rate":  (sum(r[hit_rate_f] for r in evaluated_) / n_eval_) if n_eval_ else 0.0,
        "avg_accuracy":  (sum(r[accuracy_f] for r in evaluated_) / n_eval_) if n_eval_ else 0.0,
        "avg_first_round_recall":   (sum(r.get(first_round_recall_f, 0.0) for r in evaluated_) / n_eval_) if n_eval_ else 0.0,
        "avg_first_round_hit_rate": (sum(r.get(first_round_hit_rate_f, 0.0) for r in evaluated_) / n_eval_) if n_eval_ else 0.0,
        "avg_selected": sum(r["num_selected"] for r in results) / n,
        "avg_rounds":   sum(r.get("num_rounds", 0) for r in results) / n,
        "avg_time_per_instance_sec": sum(r.get("elapsed_sec", 0) for r in results) / n,
    }
    # Retrieval-only control at the same selection budget. Reported over the same
    # evaluated set as avg_recall, so `lift` is an apples-to-apples difference.
    bl_recall_f = f"{baseline_prefix}recall"
    if evaluated_ and bl_recall_f in evaluated_[0]:
        a["baseline_k"] = evaluated_[0].get("baseline_k")
        for m_ in ("recall", "precision", "hit_rate", "accuracy"):
            a[f"avg_baseline_{m_}"] = sum(r[f"{baseline_prefix}{m_}"] for r in evaluated_) / n_eval_
        a["avg_recall_lift_over_baseline"] = a["avg_recall"] - a["avg_baseline_recall"]
        a["avg_hit_rate_lift_over_baseline"] = a["avg_hit_rate"] - a["avg_baseline_hit_rate"]

    tp = sum(r["usage"]["prompt_tokens"] for r in results)
    tc = sum(r["usage"]["completion_tokens"] for r in results)
    # reasoning is a SUBSET of completion_tokens (0 for non-reasoning models / older
    # runs that predate reasoning capture). total_tokens stays prompt+completion.
    tr = sum(r["usage"].get("reasoning_tokens", 0) for r in results)
    tcalls = sum(r["usage"]["num_calls"] for r in results)
    a["avg_prompt_tokens"] = tp / n
    a["avg_completion_tokens"] = tc / n
    a["avg_reasoning_tokens"] = tr / n
    a["avg_total_tokens"] = (tp + tc) / n
    a["avg_per_call_prompt_tokens"] = tp / tcalls if tcalls else 0
    a["avg_per_call_completion_tokens"] = tc / tcalls if tcalls else 0
    a["avg_per_call_reasoning_tokens"] = tr / tcalls if tcalls else 0
    a["total_prompt_tokens"] = tp
    a["total_completion_tokens"] = tc
    a["total_reasoning_tokens"] = tr
    a["total_tokens"] = tp + tc
    a["avg_num_calls"] = tcalls / n
    a["early_stop_count"] = sum(1 for r in results if r.get("early_stop"))
    a["confident_stop_count"] = sum(1 for r in results if r.get("confident"))
    a["stop_reason_counts"] = dict(Counter(r.get("stop_reason", "?") for r in results))
    return a, n_eval_


def build_views(all_results: List[Dict]) -> Dict:
    """Split succeeded/failed and build both GT-view aggregates."""
    failed_results = [r for r in all_results if r.get("failed")]
    results = [r for r in all_results if not r.get("failed")]

    ours_agg, ours_n_eval = build_agg(
        results, failed_results, all_results,
        view_name="locagent",
        eval_filter=lambda r: bool(r.get("ground_truth")),
        recall_f="recall", precision_f="precision",
        hit_rate_f="hit_rate", accuracy_f="accuracy",
    )

    has_spider = any("spider_ground_truth" in r for r in all_results)
    spider_agg, spider_n_eval = ({}, 0)
    if has_spider:
        spider_agg, spider_n_eval = build_agg(
            results, failed_results, all_results,
            view_name="modified_nodes_filtered_pool",
            eval_filter=lambda r: bool(r.get("spider_evaluated")),
            recall_f="spider_recall", precision_f="spider_precision",
            hit_rate_f="spider_hit_rate", accuracy_f="spider_accuracy",
            first_round_recall_f="spider_first_round_recall",
            first_round_hit_rate_f="spider_first_round_hit_rate",
            baseline_prefix="spider_baseline_",
        )

    return {
        "results": results, "failed_results": failed_results,
        "ours_agg": ours_agg, "ours_n_eval": ours_n_eval,
        "spider_agg": spider_agg, "spider_n_eval": spider_n_eval,
        "has_spider": has_spider,
    }


def log_agg(label: str, agg: Dict, n_eval: int, pipeline: str, n: int, n_failed: int, extra: str = ""):
    logger.info(f"\n=== Aggregate [{label}] ({pipeline}, {n} succeeded / "
                f"{n_eval} evaluated / {n_failed} failed{extra}) ===")
    for k, v in agg.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")


def dump_results(out_dir, tag: str, cfg, pipeline: str, all_results: List[Dict],
                 gpus: List = None) -> List:
    """Write results JSON(s). Returns the paths written.

    The spider pipeline emits both GT views, so it gets a two-file split
    (`_ours.json` / `_spider.json`); everything else writes a single file.
    """
    views = build_views(all_results)
    results, failed_results = views["results"], views["failed_results"]

    if failed_results:
        logger.warning(f"\n{'!'*80}")
        logger.warning(f"  {len(failed_results)} / {len(all_results)} instances FAILED:")
        for r in failed_results:
            err = r.get("error", {})
            logger.warning(f"    [{r['instance_id']}] {err.get('type','?')}: "
                           f"{err.get('message','?')[:200]}  (at round {err.get('round_attempted')})")
        logger.warning(f"{'!'*80}\n")

    failed_summary = [
        {
            "instance_id": r["instance_id"],
            "error": r.get("error", {}),
            "num_rounds_completed": r.get("num_rounds", 0),
            "partial_selected": r.get("selected", []),
        }
        for r in failed_results
    ]

    def _dump(path, agg_):
        with open(path, "w") as f:
            json.dump({
                "config": redact_config(cfg),
                "gpus": list(gpus or []),
                "pipeline": pipeline,
                "aggregate": agg_,
                "num_failed": len(failed_results),
                "failed_summary": failed_summary,
                "results": results,
                "failed_results": failed_results,  # includes partial trajectories
            }, f, indent=2, default=str)

    if views["has_spider"]:
        p_ours = out_dir / f"results_{tag}_ours.json"
        p_spider = out_dir / f"results_{tag}_spider.json"
        _dump(p_ours, views["ours_agg"])
        _dump(p_spider, views["spider_agg"])
        written = [p_ours, p_spider]
    else:
        p = out_dir / f"results_{tag}.json"
        _dump(p, views["ours_agg"])
        written = [p]

    return written
