# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt rendering and response parsing for the SpIDER neighbor-filter call.

Nodes are shown to the model with both a bracket label ([nbr_3]) and their full
node ID, and `_resolve` accepts either form back -- models are inconsistent about
which one they echo, and an unresolvable ID is dropped rather than guessed at.
"""

import json
import re
from typing import Dict, List

from .data_loader import InstanceData


def _build_idx_map(nodes: List[str], prefix: str = "", offset: int = 0) -> Dict[str, str]:
    """Map bracket indices, prefixed labels, and full node IDs to actual IDs."""
    m: Dict[str, str] = {}
    for i, nid in enumerate(nodes):
        idx = offset + i + 1
        m[str(idx)] = nid
        m[f"[{idx}]"] = nid
        if prefix:
            m[f"{prefix}_{i+1}"] = nid
            m[f"[{prefix}_{i+1}]"] = nid
        m[nid] = nid
    return m


def _resolve(raw_ids: List[str], idx_map: Dict[str, str]) -> tuple:
    """Resolve raw IDs -> (resolved_list, unresolved_list)."""
    resolved, unresolved = [], []
    seen = set()
    for raw in raw_ids:
        raw = raw.strip()
        nid = idx_map.get(raw) or idx_map.get(re.sub(r'[\[\]]', '', raw))
        if nid and nid not in seen:
            seen.add(nid)
            resolved.append(nid)
        elif not nid:
            unresolved.append(raw)
    return resolved, unresolved


def _parse_json_obj(response_text: str) -> dict:
    """Extract a JSON object from an LLM response (tolerates ``` fencing)."""
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _parse_selected(response_text: str) -> List[str]:
    """Extract selected node IDs from the LLM's JSON response."""
    obj = _parse_json_obj(response_text)
    sel = obj.get("selected", [])
    return sel if isinstance(sel, list) else [sel]


def _format_nodes(instance: InstanceData, nodes: List[str], prefix: str,
                  offset: int = 0) -> str:
    """Format a list of nodes with their code for the prompt."""
    parts = []
    for i, nid in enumerate(nodes):
        idx = offset + i + 1
        code = instance.get_node_code(nid)
        scores = instance.node_scores.get(nid, {})
        score_str = f"sim={scores.get('dense_similarity', 0):.4f}, bm25={scores.get('bm25_score', 0):.2f}"
        label = f"[{prefix}_{idx}]" if prefix else f"[{idx}]"
        parts.append(f"### {label} {nid}  ({score_str})\n```\n{code}\n```")
    return "\n\n".join(parts)
