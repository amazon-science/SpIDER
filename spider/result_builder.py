# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-instance result construction.

`build_result` assembles the result dict the SpIDER pipeline emits: both GT
views (LocAgent patch-overlap GT and SpIDER `modified_nodes` GT), the no-LLM
top-L dense-retrieval baseline at the same budget, token usage, and per-call
stats. summarize.py consumes this schema directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def score_set(selected: set, gt: set, prefix: str = "") -> Dict:
    """recall / precision / hit_rate / accuracy of `selected` against `gt`."""
    hits = selected & gt
    return {
        f"{prefix}recall": len(hits) / len(gt) if gt else 0.0,
        f"{prefix}precision": len(hits) / len(selected) if selected else 0.0,
        f"{prefix}hit_rate": 1.0 if hits else 0.0,
        f"{prefix}accuracy": 1.0 if gt and hits == gt else 0.0,
    }


def build_result(
    instance,
    cfg,
    *,
    selected: List[str],
    first_round_selected: Optional[List[str]] = None,
    usage: Dict,
    rounds: Optional[List[Dict]] = None,
    num_rounds: Optional[int] = None,
    stop_reason: str = "completed",
    early_stop: bool = False,
    failed: bool = False,
    error: Optional[Dict] = None,
    confident: bool = False,
    total_seen: Optional[int] = None,
    extra: Optional[Dict] = None,
    config_extra: Optional[Dict] = None,
) -> Dict:
    """Assemble the standard result dict.

    Args:
      selected  : final predicted node ids (already truncated to the budget).
      first_round_selected : the selection after the first round/step (defaults to
                  `selected` for single-shot pipelines) -- feeds first_round_* metrics.
      usage     : {prompt_tokens, completion_tokens, reasoning_tokens, num_calls}.
      rounds    : per-round/step trajectory log (or None).
      extra     : pipeline-specific fields to merge in (e.g. total_nbr_selected).
      config_extra : extra keys for the serialized `config` block.
    """
    final = list(selected)
    sel = set(final)
    if first_round_selected is None:
        first_round_selected = final

    gt = set(instance.gt_nodes)
    hits = sel & gt
    spider_gt = set(instance.spider_gt_nodes)
    spider_hits = sel & spider_gt

    first_set = set(first_round_selected)
    first_hits = first_set & gt
    first_spider_hits = first_set & spider_gt

    # No-LLM retrieval control: top-L of whatever ranking the pipeline used
    # (instance.ranked_nodes), scored against both GT views. Same budget as `final`.
    baseline = set(instance.ranked_nodes[: cfg.L])
    baseline_scores = score_set(baseline, gt, prefix="baseline_")
    baseline_scores.update(score_set(baseline, spider_gt, prefix="spider_baseline_"))

    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    reasoning = usage.get("reasoning_tokens", 0)
    num_calls = usage.get("num_calls", 0)

    result = {
        "instance_id": instance.instance_id,
        "node_type": cfg.node_type,
        "selected": final,
        "num_selected": len(final),
        "ground_truth": list(gt),
        "hits": list(hits),
        "recall": len(hits) / len(gt) if gt else 0.0,
        "precision": len(hits) / len(sel) if sel else 0.0,
        "hit_rate": 1.0 if hits else 0.0,
        "accuracy": 1.0 if gt and hits == gt else 0.0,
        "first_round_recall": len(first_hits) / len(gt) if gt else 0.0,
        "first_round_hit_rate": 1.0 if first_hits else 0.0,
        "first_round_accuracy": 1.0 if gt and first_hits == gt else 0.0,
        "baseline_k": cfg.L,
        "baseline_selected": sorted(baseline),
        **baseline_scores,
        "spider_ground_truth": list(spider_gt),
        "spider_hits": list(spider_hits),
        "spider_evaluated": bool(spider_gt),
        "spider_recall": len(spider_hits) / len(spider_gt) if spider_gt else 0.0,
        "spider_precision": len(spider_hits) / len(sel) if sel else 0.0,
        "spider_hit_rate": 1.0 if spider_hits else 0.0,
        "spider_accuracy": 1.0 if spider_gt and spider_hits == spider_gt else 0.0,
        "spider_first_round_recall": len(first_spider_hits) / len(spider_gt) if spider_gt else 0.0,
        "spider_first_round_hit_rate": 1.0 if first_spider_hits else 0.0,
        "spider_first_round_accuracy": 1.0 if spider_gt and first_spider_hits == spider_gt else 0.0,
        "num_rounds": num_rounds if num_rounds is not None else (len(rounds) if rounds else 0),
        "early_stop": early_stop,
        "stop_reason": stop_reason,
        "confident": confident,
        "total_seen": total_seen if total_seen is not None else len(sel),
        "failed": failed,
        "error": error,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,      # includes reasoning
            "reasoning_tokens": reasoning,
            "total_tokens": prompt + completion,
            "num_calls": num_calls,
        },
        "per_call_prompt_tokens": prompt / num_calls if num_calls else 0,
        "per_call_completion_tokens": completion / num_calls if num_calls else 0,
        "per_call_reasoning_tokens": reasoning / num_calls if num_calls else 0,
        "rounds": rounds or [],
        "config": {
            "K": cfg.K, "L": cfg.L, "N": cfg.N, "C": cfg.C,
            "max_steps": cfg.max_steps, "max_code_chars": cfg.max_code_chars,
            "model": cfg.model, "ranking_type": cfg.ranking_type,
            "edge_type": cfg.edge_type, "node_type": cfg.node_type,
            "bfs_depth": cfg.bfs_depth,
            "edges": cfg.edge_list(), "edge_depth_map": cfg.edge_depth_map(),
            "edge_slug": cfg.edge_slug(),
            **(config_extra or {}),
        },
    }
    if extra:
        result.update(extra)
    return result
