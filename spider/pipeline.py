# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SpIDER: deterministic top-C centers + restrictive BFS neighbor filtering.

Reference implementation of the localization loop in arXiv:2512.16956.

The LLM makes exactly one kind of decision -- "does this graph neighbor itself
need to change?" -- and everything else is deterministic:

  - Centers are the top-C of the current K window (no LLM call).
  - Step B asks the LLM, once per center, to restrictively filter that center's
    BFS neighborhood (C calls per round).
  - The modified window is assembled by rule: each selected neighbor is inserted
    directly below its center, and an equal number of bottom-ranked nodes drop
    out to keep the window at size K.

Per round: exactly C LLM calls. The loop runs up to `max_steps` rounds, or stops
early when the ranked list is exhausted.
"""

import logging
import traceback
from typing import Dict, List

from .config import SelectorConfig
from .data_loader import InstanceData
from .prompt_format import (
    _build_idx_map, _resolve, _parse_selected, _format_nodes,
)
from .result_builder import build_result

logger = logging.getLogger(__name__)


def _edge_phrase(cfg: SelectorConfig) -> str:
    """e.g. "'contains' edges (up to 4 hops)" or, for multiple edge types,
    "'contains' edges (up to 4 hops) or 'invokes' edges (up to 2 hops)"."""
    m = cfg.edge_depth_map()
    parts = [f"'{e}' edges (up to {m[e]} hops)" for e in sorted(m)]
    return " or ".join(parts)


NEIGHBOR_SYSTEM_RESTRICTIVE = """You are a code localization agent. You are examining the graph neighborhood of a {node_type} to find related {node_type}s that MUST be modified to fix a GitHub issue.

You will see:
- The center {node_type}
- Its graph neighbors (connected via {edge_phrase})

BE VERY RESTRICTIVE. Only select neighbors where you have HIGH CONFIDENCE that the code MUST be modified to fix the issue. When in doubt, do NOT select. It is better to return an empty list than to include a neighbor that probably does not need changes.

Do NOT include the center itself. Do NOT include neighbors that are merely related, called by, or similar to code in the issue — only include neighbors whose own code must change.

Respond with ONLY a JSON object:
{{"selected": ["node_id_1", "node_id_2", ...]}}

If no neighbor clearly needs changes, respond with:
{{"selected": []}}"""


def run_step_b_restrictive(instance: InstanceData, cfg: SelectorConfig, client,
                           centers: List[str], skip: set) -> Dict:
    """For each center, LLM filters its 4-hop BFS neighbors restrictively.

    One call per center. The system prompt enforces high-confidence selection.
    """
    edge_phrase = _edge_phrase(cfg)
    system = NEIGHBOR_SYSTEM_RESTRICTIVE.format(
        node_type=cfg.node_type, edge_phrase=edge_phrase)

    per_center = []

    for ci, center_id in enumerate(centers):
        neighbors = instance.get_filtered_neighbors([center_id], 0, skip=skip)

        # Dense edges (e.g. `invokes`) can make the BFS cover most of the
        # N-window, overflowing the model's context. Neighbors are rank-ordered,
        # so truncating keeps the most promising ones.
        n_before = len(neighbors)
        cap = cfg.max_neighbors_per_center
        if cap and n_before > cap:
            neighbors = neighbors[:cap]
            logger.warning(f"[{instance.instance_id}] center {center_id}: "
                           f"truncated {n_before} neighbors -> {cap}")

        parts = [f"## GitHub Issue\n{instance.problem_statement}"]
        parts.append(f"\n## Center {cfg.node_type} ({ci+1}/{len(centers)})")
        parts.append(_format_nodes(instance, [center_id], prefix="center", offset=0))
        if neighbors:
            parts.append(f"\n## Neighbors ({len(neighbors)} {cfg.node_type}s via {edge_phrase})")
            parts.append(_format_nodes(instance, neighbors, prefix="nbr", offset=0))
        else:
            parts.append(f"\n## Neighbors\n  (no unseen neighbors found)")
        parts.append(f"\nSelect ONLY the {cfg.node_type}s that you are highly confident MUST be modified to fix the issue.")
        user_msg = "\n\n".join(parts)

        idx_map = _build_idx_map([center_id], prefix="center", offset=0)
        idx_map.update(_build_idx_map(neighbors, prefix="nbr", offset=0))

        call_error = None
        try:
            resp = client.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ])
            raw = resp.content
            usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                     "reasoning_tokens": resp.reasoning_tokens}
        except Exception as e:
            logger.error(f"[{instance.instance_id}] Step B center {center_id} failed: {e}")
            raw = ""
            usage = {}
            # An empty selection here is indistinguishable from a confident "no
            # neighbors need changing", so record the failure explicitly --
            # otherwise a context-length overflow silently degrades the run to
            # the plain ranking baseline.
            call_error = f"{type(e).__name__}: {e}"

        # A reasoning model can burn its whole max_output_tokens budget on
        # reasoning and return no text. That parses to an empty selection, which
        # is indistinguishable from a confident "nothing needs changing" -- flag it.
        empty_response = call_error is None and not raw.strip()
        if empty_response:
            logger.warning(f"[{instance.instance_id}] center {center_id}: "
                           f"empty response (raise --max_tokens?)")

        raw_selected = _parse_selected(raw)
        resolved, unresolved = _resolve(raw_selected, idx_map)

        # Restrictive: exclude the center itself, keep only visible neighbors
        nbr_set = set(neighbors)
        valid = [nid for nid in resolved if nid in nbr_set]

        per_center.append({
            "center": center_id,
            "neighbors_shown": neighbors,
            "neighbors_found": n_before,
            "neighbors_truncated": n_before - len(neighbors),
            "raw_selected": raw_selected,
            "resolved": resolved,
            "unresolved": unresolved,
            "selected": valid,
            "raw_response": raw,
            "usage": usage,
            "call_error": call_error,
            "empty_response": empty_response,
        })

    return {"per_center": per_center}


def build_modified_window(window: List[str], centers: List[str],
                          per_center: List[Dict], K: int) -> List[str]:
    """Insert each center's selected neighbors immediately below the center,
    then truncate to K (dropping equal-count bottom-ranked nodes).
    """
    center_to_selected: Dict[str, List[str]] = {}
    for pc in per_center:
        center_to_selected.setdefault(pc["center"], []).extend(pc["selected"])

    modified: List[str] = []
    seen: set = set()
    for nid in window:
        if nid not in seen:
            modified.append(nid)
            seen.add(nid)
        for nbr in center_to_selected.get(nid, []):
            if nbr not in seen:
                modified.append(nbr)
                seen.add(nbr)

    return modified[:K]


def run_spider_pipeline(instance: InstanceData, cfg: SelectorConfig, client) -> Dict:
    """Run the SpIDER loop for one instance.

    Round r:
      - Fetch K fresh unseen ranked nodes → window
      - centers = window[:C]  (deterministic, no LLM)
      - Step B: LLM filters BFS neighbors restrictively per center (C calls)
      - Build modified window: insert selected neighbors below centers, truncate to K
      - Update current_L: prior current_L items first, then fill from modified window

    Stop when: max_steps reached OR ranked list exhausted.
    Total calls per round: C.
    """
    max_rounds = max(1, cfg.max_steps)
    current_L: List[str] = []
    offset = 0
    seen: set = set()
    rounds_log = []

    total_prompt = 0
    total_completion = 0
    total_reasoning = 0
    num_calls = 0
    num_call_errors = 0
    num_empty_responses = 0
    total_nbr_truncated = 0
    early_stop = False
    stop_reason = "budget_exhausted"
    error_info = None

    try:
        for round_idx in range(max_rounds):
            window = instance.get_window(offset, skip=seen)
            if not window:
                early_stop = True
                stop_reason = "ranked_list_exhausted"
                break

            C = min(cfg.C, len(window))
            centers = window[:C]

            seen.update(window)

            step_b = run_step_b_restrictive(instance, cfg, client, centers, skip=seen)
            for pc in step_b["per_center"]:
                total_prompt += pc["usage"].get("prompt_tokens", 0)
                total_completion += pc["usage"].get("completion_tokens", 0)
                total_reasoning += pc["usage"].get("reasoning_tokens", 0)
                num_calls += 1
                if pc["call_error"]:
                    num_call_errors += 1
                if pc["empty_response"]:
                    num_empty_responses += 1
                total_nbr_truncated += pc["neighbors_truncated"]
                seen.update(pc["neighbors_shown"])

            modified_window = build_modified_window(window, centers, step_b["per_center"], cfg.K)

            new_current_L: List[str] = []
            for nid in current_L:
                if nid not in new_current_L:
                    new_current_L.append(nid)
            for nid in modified_window:
                if nid not in new_current_L:
                    new_current_L.append(nid)
                if len(new_current_L) >= cfg.L:
                    break
            new_L = new_current_L[:cfg.L]

            all_selected_neighbors = [
                nbr for pc in step_b["per_center"] for nbr in pc["selected"]
            ]

            rounds_log.append({
                "round": round_idx,
                "offset": offset,
                "window": window,
                "centers": centers,
                "step_b_per_center": step_b["per_center"],
                "step_b_all_selected": all_selected_neighbors,
                "modified_window": modified_window,
                "current_L_before": current_L,
                "new_L": new_L,
            })

            nbr_shown = sum(len(pc["neighbors_shown"]) for pc in step_b["per_center"])
            logger.info(f"[{instance.instance_id}] Round {round_idx}: "
                        f"centers={len(centers)} nbr_shown={nbr_shown} "
                        f"nbr_selected={len(all_selected_neighbors)} "
                        f"errors={num_call_errors} empty={num_empty_responses} "
                        f"|modified_window|={len(modified_window)}")

            current_L = new_L
            offset += cfg.K
    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "round_attempted": len(rounds_log),
            "traceback": traceback.format_exc(),
        }
        stop_reason = "error"
        logger.error(f"[{instance.instance_id}] Pipeline failed at round {len(rounds_log)}: "
                     f"{type(e).__name__}: {e}")

    final = current_L[:cfg.L]
    first_L = rounds_log[0]["new_L"] if rounds_log else []
    total_nbr_selected = sum(len(r["step_b_all_selected"]) for r in rounds_log)

    return build_result(
        instance, cfg,
        selected=final,
        first_round_selected=first_L,
        usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion,
               "reasoning_tokens": total_reasoning, "num_calls": num_calls},
        rounds=rounds_log,
        stop_reason=stop_reason,
        early_stop=early_stop,
        failed=error_info is not None,
        error=error_info,
        confident=False,          # SpIDER has no confidence flag (no refinement stage)
        total_seen=len(seen),
        extra={
            "total_nbr_selected": total_nbr_selected,
            "total_nbr_truncated": total_nbr_truncated,
            # >0 means some Step B calls errored (often context overflow) and were
            # treated as empty selections -- those rounds fell back to plain ranking.
            "num_call_errors": num_call_errors,
            # >0 means some Step B calls returned no text (reasoning models can spend
            # the whole max_output_tokens budget on reasoning). Same silent fallback.
            "num_empty_responses": num_empty_responses,
        },
    )
