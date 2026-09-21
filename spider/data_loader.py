# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load precomputed rankings, repository graphs, and code for a single instance."""

import ast as _ast
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx

from .config import SelectorConfig


# Tree-sitter node kinds used by SWE-PolyBench's `modified_nodes` field
# across the four languages.
_FUNC_KINDS = {
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "constructor_definition", "constructor_declaration",
    "arrow_function", "function_expression", "generator_function_declaration",
    "function", "method",
}
_CLASS_KINDS = {
    "class_definition", "class_declaration", "interface_declaration",
    "enum_declaration", "trait_declaration",
}


def _parse_modified_nodes(mn_str) -> List[Tuple[str, str, str]]:
    """Parse SWE-PolyBench `modified_nodes` -> list of (file, qualified_name, leaf_kind).

    leaf_kind in {"function", "class"}. Language-agnostic; recognizes tree-sitter
    node kinds from the python/java/javascript/typescript grammars.
    """
    if not mn_str or mn_str == "[]":
        return []
    try:
        items = _ast.literal_eval(mn_str) if isinstance(mn_str, str) else mn_str
    except Exception:
        return []
    out: List[Tuple[str, str, str]] = []
    for s in items:
        parts = s.split("->")
        if len(parts) < 2:
            continue
        fp = parts[0]
        qual: List[str] = []
        leaf: Optional[str] = None
        for p in parts[1:]:
            if p in ("module", "program"):
                continue
            if ":" in p:
                kind, name = p.split(":", 1)
                if kind in _FUNC_KINDS:
                    qual.append(name); leaf = "function"
                elif kind in _CLASS_KINDS:
                    qual.append(name); leaf = "class"
                else:
                    qual.append(name)
        if leaf and qual:
            out.append((fp, ".".join(qual), leaf))
    return out


def _build_spider_gt(
    modnodes_str, full_ranked: Dict[str, dict]
) -> Tuple[List[str], List[str], List[str]]:
    """Build (gt_files, gt_classes, gt_funcs) from `modified_nodes`, filtered to pool.

    `full_ranked` is the all-type ranked dict for the instance (e.g. `dense_ranked`).
    A modnode entry is kept only if the constructed node ID is present as a key in
    `full_ranked` (i.e. the candidate retrieval pool). File-level GT is the set of
    file paths that contain at least one function/class modnode, intersected with
    file-type entries in the pool.

    Instances whose `modified_nodes` field is absent (e.g. SWE-bench Verified) get
    empty SpIDER GT and are reported only under the LocAgent GT view.
    """
    parsed = _parse_modified_nodes(modnodes_str)
    if not parsed:
        return [], [], []
    pool = set(full_ranked.keys())
    cand_funcs = {f"{fp}:{q}" for fp, q, k in parsed if k == "function"}
    cand_classes = {f"{fp}:{q}" for fp, q, k in parsed if k == "class"}
    cand_files = {fp for fp, _, _ in parsed}
    return (
        sorted(cand_files & pool),
        sorted(cand_classes & pool),
        sorted(cand_funcs & pool),
    )


class InstanceData:
    """Preprocessed data for a single issue instance, filtered by node type."""

    def __init__(self, instance_id: str, issue: dict, graph: nx.MultiDiGraph, cfg: SelectorConfig):
        self.instance_id = instance_id
        self.problem_statement: str = issue["problem_statement"]
        self.graph = graph
        self.cfg = cfg

        # Ground truth (for evaluation after the run) -- LocAgent's patch-line-overlap GT
        self.gt_files: List[str] = issue.get("gt_files", [])
        self.gt_classes: List[str] = issue.get("gt_classes", [])
        self.gt_funcs: List[str] = issue.get("gt_funcs", [])

        # Build the type-filtered ranked list from the full ranking.
        full_ranked: Dict[str, dict] = issue[cfg.ranking_key()]
        self.ranked_nodes: List[str] = []
        self.node_scores: Dict[str, dict] = {}
        for node_id, scores in full_ranked.items():
            ntype = graph.nodes[node_id].get("type") if node_id in graph.nodes else None
            if ntype == cfg.node_type:
                self.ranked_nodes.append(node_id)
                self.node_scores[node_id] = scores

        # SpIDER-style GT: derived from the SWE-PolyBench `modified_nodes` field
        # and intersected with the full candidate pool (all-type ranked).
        modnodes_str = (issue.get("issue_fields", {}) or {}).get("modified_nodes")
        self.spider_gt_files, self.spider_gt_classes, self.spider_gt_funcs = \
            _build_spider_gt(modnodes_str, full_ranked)

    def get_node_code(self, node_id: str) -> str:
        """Return (possibly truncated) code for a node."""
        code = self.graph.nodes[node_id].get("code", "") if node_id in self.graph.nodes else ""
        if len(code) > self.cfg.max_code_chars:
            code = code[: self.cfg.max_code_chars] + "\n... [truncated]"
        return code

    def get_window(self, offset: int, skip: set = None) -> List[str]:
        """Return K unseen node IDs from the ranked list starting at offset.

        If skip is provided, skips nodes already seen and continues scanning
        to fill the window with K fresh nodes.
        """
        if not skip:
            return self.ranked_nodes[offset: offset + self.cfg.K]
        result = []
        for nid in self.ranked_nodes[offset:]:
            if nid not in skip:
                result.append(nid)
                if len(result) >= self.cfg.K:
                    break
        return result

    def get_neighbor_candidates(self, offset: int, skip: set = None) -> List[str]:
        """Return up to N unseen node IDs after the window as the neighbor filter pool."""
        start = offset + self.cfg.K
        if not skip:
            return self.ranked_nodes[start: start + self.cfg.N]
        result = []
        for nid in self.ranked_nodes[start:]:
            if nid not in skip:
                result.append(nid)
                if len(result) >= self.cfg.N:
                    break
        return result

    def get_filtered_neighbors(self, center_ids: List[str], offset: int, skip: set = None) -> List[str]:
        """BFS neighbors of center_ids, filtered to the N-window in the ranked list.

        1. Collect BFS neighbors of each center node from the graph.
        2. Keep only those that appear in the neighbor candidate pool.
        3. Exclude nodes in the skip set (already seen).
        """
        neighbor_window = set(self.get_neighbor_candidates(offset, skip=skip))
        if not neighbor_window:
            return []

        all_bfs = set()
        for cid in center_ids:
            all_bfs.update(self._bfs_from_node(cid))

        # Intersect with the neighbor window and preserve rank order.
        filtered = [n for n in self.get_neighbor_candidates(offset, skip=skip) if n in all_bfs]
        return filtered

    def adj_for_edge(self, etype: str) -> Dict[str, set]:
        """Undirected adjacency for ONE edge type, cached per etype.

        Built once per (instance, etype). Eliminates the per-frontier-node
        `g.out_edges + g.in_edges + filter-by-type` overhead in `_bfs_from_node`,
        which is the main BFS cost for dense edges like `invokes`.
        """
        cache = getattr(self, "_adj_cache", None)
        if cache is None:
            cache = {}
            self._adj_cache = cache
        if etype not in cache:
            from collections import defaultdict
            adj: Dict[str, set] = defaultdict(set)
            for u, v, edata in self.graph.edges(data=True):
                if edata.get("type") == etype:
                    adj[u].add(v)
                    adj[v].add(u)
            cache[etype] = dict(adj)
        return cache[etype]

    def _bfs_from_node(self, node_id: str, depth: Optional[int] = None) -> set:
        """Undirected BFS neighbors of node_id, filtered to cfg.node_type.

        Expands each edge type in `cfg.edge_list()` independently from the seed,
        at its own depth from `cfg.edge_depth_map()`, then unions the results.
        Mixed-type paths (2 contains-hops then 2 invokes-hops) are therefore NOT
        traversed -- that is what lets `invokes` be capped at depth 2 while
        `contains` stays at depth 4 without one explosion swamping the other.

        `depth`, if given, overrides every edge's depth uniformly.
        """
        g = self.graph
        if node_id not in g.nodes:
            return set()

        edge_depths = self.cfg.edge_depth_map()
        if depth is not None:
            edge_depths = {e: int(depth) for e in edge_depths}

        visited_all: set = set()
        for etype, d in edge_depths.items():
            if d <= 0:
                continue
            adj = self.adj_for_edge(etype)
            visited: set = set()
            frontier: set = {node_id}
            for _ in range(d):
                next_frontier: set = set()
                for n in frontier:
                    for nei in adj.get(n, ()):
                        if nei not in visited:
                            next_frontier.add(nei)
                visited.update(next_frontier)
                frontier = next_frontier
            visited_all.update(visited)

        return {n for n in visited_all if g.nodes.get(n, {}).get("type") == self.cfg.node_type}

    @property
    def gt_nodes(self) -> List[str]:
        """LocAgent ground truth nodes for the configured node type."""
        mapping = {"file": self.gt_files, "class": self.gt_classes, "function": self.gt_funcs}
        return mapping.get(self.cfg.node_type, [])

    @property
    def spider_gt_nodes(self) -> List[str]:
        """SpIDER-style ground truth (modnodes n candidate pool) for the node type."""
        mapping = {
            "file": self.spider_gt_files,
            "class": self.spider_gt_classes,
            "function": self.spider_gt_funcs,
        }
        return mapping.get(self.cfg.node_type, [])


def load_scores(cfg: SelectorConfig) -> dict:
    """Load the precomputed issue-node scores pickle."""
    with open(cfg.scores_path, "rb") as f:
        return pickle.load(f)


def _resolve_graph_file(
    graph_file: str,
    instance_id: Optional[str] = None,
    cfg: Optional[SelectorConfig] = None,
) -> str:
    """Resolve the per-instance graph pickle.

    Score pickles store the absolute `graph_file` path of the machine that built
    them, which almost never exists on a different host. Try the stored path
    first, then fall back to `{cfg.graphs_dir}/{instance_id}.pkl` -- which is what
    `--graphs_dir` is for.
    """
    if graph_file and Path(graph_file).exists():
        return graph_file
    if cfg is not None and instance_id is not None and getattr(cfg, "graphs_dir", None):
        composed = Path(cfg.graphs_dir) / f"{instance_id}.pkl"
        if composed.exists():
            return str(composed)
    return graph_file


def load_instance(instance_id: str, scores_data: dict, cfg: SelectorConfig) -> InstanceData:
    """Build InstanceData for a single issue."""
    issue = scores_data["issues"][instance_id]
    graph_file = _resolve_graph_file(issue.get("graph_file"), instance_id=instance_id, cfg=cfg)
    with open(graph_file, "rb") as f:
        graph = pickle.load(f)
    return InstanceData(instance_id, issue, graph, cfg)
