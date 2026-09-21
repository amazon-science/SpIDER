# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: attach the BFS-expanded candidate pool to the score pickle.

For each issue and each edge type present in its graph, takes the top-k ranked
nodes as centers, BFS-expands them to `--depth` hops, keeps only neighbors that
are within the top-`--filter_top_n` of the ranking, and stores the resulting
ordered list under `bfs_{depth}_{edge_type}_{dense,bm25}_ranked[_{node_type}]`.

This stage is what turns `*_issue_node_scores.pkl` into
`*_issue_node_scores_bfs4_filtered500.pkl`. It only ADDS fields; `dense_ranked`,
`bm25_ranked` and the ground truth are passed through untouched.

Note on the SpIDER runner: the pipeline re-runs BFS live against the graph at
its own configured depth, so it reads `dense_ranked` but not the `bfs_*` fields
this stage writes. Stage 3 is therefore optional for running SpIDER itself -- it
is included because the released artifacts were produced from its output, and
because the `bfs_*` fields are what the retrieval-ceiling analyses use.

Usage:
    python -m data_prep.add_filtered_bfs_rankings \
        --scores_path data/node_embeddings/swepolybench/test_python_issue_node_scores.pkl \
        --output_path data/node_embeddings/swepolybench/test_python_issue_node_scores_bfs4_filtered500.pkl \
        --graphs_dir  data/SpIDER-Bench/SWE-PolyBench/python \
        --k_values 1,3,5,10,20,50,100,500 --depth 4 --filter_top_n 500 --max_workers 64
"""

import argparse
import json
import os.path as osp
import pickle
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tqdm.auto import tqdm


TARGET_NODE_TYPES = ("file", "class", "function")


def load_issue_scores(path: str) -> dict:
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_issue_scores(obj: dict, path: str):
    out_dir = osp.dirname(path)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def parse_k_values(k_values: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in k_values.split(",") if x.strip()))
    if not vals:
        raise ValueError("k_values must contain at least one integer")
    if any(k <= 0 for k in vals):
        raise ValueError("all k_values must be > 0")
    return vals


def build_undirected_adjacency_by_edge_type(graph) -> Dict[str, Dict[str, Set[str]]]:
    by_type: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for u, v, _, attrs in graph.edges(keys=True, data=True):
        etype = attrs.get("type")
        if not isinstance(etype, str) or not etype:
            continue
        su, sv = str(u), str(v)
        by_type[etype][su].add(sv)
        by_type[etype][sv].add(su)
    return by_type


def bfs_neighbors(
    center: str,
    center_node_type: Optional[str],
    adjacency: Dict[str, Set[str]],
    node_types: Dict[str, str],
    depth: int,
    same_type_only: bool,
) -> List[str]:
    if same_type_only and center_node_type is None:
        return []

    q = deque([(center, 0)])
    seen = {center}
    out = []

    while q:
        cur, d = q.popleft()
        if d == depth:
            continue
        for nei in adjacency.get(cur, set()):
            if nei in seen:
                continue
            seen.add(nei)
            q.append((nei, d + 1))
            if same_type_only and node_types.get(nei) != center_node_type:
                continue
            out.append(nei)
    return out


def ordered_union(a: Sequence[str], b: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in list(a) + list(b):
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def append_unique_in_order(base: List[str], extra: Sequence[str]) -> List[str]:
    seen = set(base)
    out = list(base)
    for x in extra:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def top_k_centers_by_type(
    ranked_node_ids: Sequence[str],
    node_types: Dict[str, str],
    max_k: int,
) -> Dict[str, List[str]]:
    by_type: Dict[str, List[str]] = {t: [] for t in TARGET_NODE_TYPES}
    for nid in ranked_node_ids:
        ntype = node_types.get(nid)
        if ntype not in by_type:
            continue
        if len(by_type[ntype]) >= max_k:
            continue
        by_type[ntype].append(nid)
    return by_type


def compute_bfs_fields_for_issue(
    issue_data: dict,
    k_values: Sequence[int],
    depth: int,
    max_workers: int,
    same_type_only: bool,
    filter_top_n: int,
    show_progress: bool,
    issue_id: str,
    graphs_dir: str = "",
) -> dict:
    if graphs_dir:
        graph_file = osp.join(graphs_dir, f"{issue_id}.pkl")
    else:
        graph_file = issue_data.get("graph_file")
    if not graph_file or not osp.exists(graph_file):
        return {
            "skipped": True,
            "reason": f"graph_file missing: {graph_file}",
            "updated_issue": issue_data,
        }

    with open(graph_file, "rb") as f:
        graph = pickle.load(f)

    node_types = {str(nid): attrs.get("type") for nid, attrs in graph.nodes(data=True)}
    adj_by_etype = build_undirected_adjacency_by_edge_type(graph)
    edge_types = sorted(adj_by_etype.keys())

    dense_ranked = issue_data.get("dense_ranked", {}) or {}
    bm25_ranked = issue_data.get("bm25_ranked", {}) or {}
    dense_ids_full = [str(x) for x in dense_ranked.keys()]
    bm25_ids_full = [str(x) for x in bm25_ranked.keys()]
    dense_filter_ids = set(dense_ids_full[:filter_top_n]) if filter_top_n > 0 else set(dense_ids_full)
    bm25_filter_ids = set(bm25_ids_full[:filter_top_n]) if filter_top_n > 0 else set(bm25_ids_full)

    if not dense_ids_full and not bm25_ids_full:
        issue_data["bfs_edge_types"] = edge_types
        issue_data["bfs_depth"] = depth
        issue_data["bfs_k_values"] = list(k_values)
        return {"skipped": False, "updated_issue": issue_data}

    max_k = max(k_values)
    dense_centers = dense_ids_full[:max_k]
    bm25_centers = bm25_ids_full[:max_k]
    dense_centers_by_type = top_k_centers_by_type(dense_ids_full, node_types, max_k)
    bm25_centers_by_type = top_k_centers_by_type(bm25_ids_full, node_types, max_k)

    # Precompute BFS neighbors for each center exactly once per edge type.
    bfs_cache_dense: Dict[str, Dict[str, List[str]]] = {etype: {} for etype in edge_types}
    bfs_cache_bm25: Dict[str, Dict[str, List[str]]] = {etype: {} for etype in edge_types}

    def precompute_for_centers(
        centers: Sequence[str],
        cache: Dict[str, Dict[str, List[str]]],
        rank_name: str,
    ):
        etype_iter = tqdm(
            edge_types,
            desc=f"{issue_id} {rank_name} edge-types",
            leave=False,
            disable=not show_progress,
        )
        for etype in etype_iter:
            adjacency = adj_by_etype.get(etype, {})
            lock = Lock()

            def worker(center: str):
                ctype = node_types.get(center)
                neigh = bfs_neighbors(
                    center=center,
                    center_node_type=ctype,
                    adjacency=adjacency,
                    node_types=node_types,
                    depth=depth,
                    same_type_only=same_type_only,
                )
                with lock:
                    cache[etype][center] = neigh

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(
                    tqdm(
                        ex.map(worker, centers),
                        total=len(centers),
                        desc=f"{issue_id} {rank_name}:{etype}",
                        leave=False,
                        disable=not show_progress,
                    )
                )

    precompute_for_centers(dense_centers, bfs_cache_dense, rank_name="dense")
    precompute_for_centers(bm25_centers, bfs_cache_bm25, rank_name="bm25")

    for k in k_values:
        top_dense = dense_centers[:k]
        top_bm25 = bm25_centers[:k]

        for etype in edge_types:
            dense_with_bfs = list(top_dense)
            for c in top_dense:
                filtered_neigh = [n for n in bfs_cache_dense[etype].get(c, []) if n in dense_filter_ids]
                dense_with_bfs = append_unique_in_order(dense_with_bfs, filtered_neigh)

            bm25_with_bfs = list(top_bm25)
            for c in top_bm25:
                filtered_neigh = [n for n in bfs_cache_bm25[etype].get(c, []) if n in bm25_filter_ids]
                bm25_with_bfs = append_unique_in_order(bm25_with_bfs, filtered_neigh)

            union_with_bfs = ordered_union(dense_with_bfs, bm25_with_bfs)

            dense_field = f"bfs_{depth}_{etype}_dense_ranked"
            bm25_field = f"bfs_{depth}_{etype}_bm25_ranked"
            union_field = f"bfs_{depth}_{etype}_union"

            if dense_field not in issue_data:
                issue_data[dense_field] = {}
            if bm25_field not in issue_data:
                issue_data[bm25_field] = {}
            if union_field not in issue_data:
                issue_data[union_field] = {}

            issue_data[dense_field][str(k)] = dense_with_bfs
            issue_data[bm25_field][str(k)] = bm25_with_bfs
            issue_data[union_field][str(k)] = union_with_bfs

            issue_data[f"{dense_field}_size"] = issue_data.get(f"{dense_field}_size", {})
            issue_data[f"{bm25_field}_size"] = issue_data.get(f"{bm25_field}_size", {})
            issue_data[f"{union_field}_size"] = issue_data.get(f"{union_field}_size", {})

            issue_data[f"{dense_field}_size"][str(k)] = len(dense_with_bfs)
            issue_data[f"{bm25_field}_size"][str(k)] = len(bm25_with_bfs)
            issue_data[f"{union_field}_size"][str(k)] = len(union_with_bfs)

            # Per-type BFS lists: top-k anchors by node type + their BFS neighbors.
            for ntype in TARGET_NODE_TYPES:
                top_dense_t = dense_centers_by_type[ntype][:k]
                top_bm25_t = bm25_centers_by_type[ntype][:k]

                dense_t_with_bfs = list(top_dense_t)
                for c in top_dense_t:
                    filtered_neigh = [n for n in bfs_cache_dense[etype].get(c, []) if n in dense_filter_ids]
                    dense_t_with_bfs = append_unique_in_order(
                        dense_t_with_bfs, filtered_neigh
                    )

                bm25_t_with_bfs = list(top_bm25_t)
                for c in top_bm25_t:
                    filtered_neigh = [n for n in bfs_cache_bm25[etype].get(c, []) if n in bm25_filter_ids]
                    bm25_t_with_bfs = append_unique_in_order(
                        bm25_t_with_bfs, filtered_neigh
                    )

                union_t_with_bfs = ordered_union(dense_t_with_bfs, bm25_t_with_bfs)

                dense_t_field = f"bfs_{depth}_{etype}_dense_ranked_{ntype}"
                bm25_t_field = f"bfs_{depth}_{etype}_bm25_ranked_{ntype}"
                union_t_field = f"bfs_{depth}_{etype}_union_{ntype}"

                issue_data.setdefault(dense_t_field, {})
                issue_data.setdefault(bm25_t_field, {})
                issue_data.setdefault(union_t_field, {})

                issue_data[dense_t_field][str(k)] = dense_t_with_bfs
                issue_data[bm25_t_field][str(k)] = bm25_t_with_bfs
                issue_data[union_t_field][str(k)] = union_t_with_bfs

                issue_data.setdefault(f"{dense_t_field}_size", {})
                issue_data.setdefault(f"{bm25_t_field}_size", {})
                issue_data.setdefault(f"{union_t_field}_size", {})

                issue_data[f"{dense_t_field}_size"][str(k)] = len(dense_t_with_bfs)
                issue_data[f"{bm25_t_field}_size"][str(k)] = len(bm25_t_with_bfs)
                issue_data[f"{union_t_field}_size"][str(k)] = len(union_t_with_bfs)

    issue_data["bfs_edge_types"] = edge_types
    issue_data["bfs_depth"] = depth
    issue_data["bfs_k_values"] = list(k_values)

    return {"skipped": False, "updated_issue": issue_data}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment issue-node score files with BFS-expanded dense/bm25/union retrieval sets "
            "per edge type and per top-k."
        )
    )
    parser.add_argument("--scores_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--graphs_dir",
        type=str,
        default="",
        help="If set, resolve graph files as {graphs_dir}/{instance_id}.pkl instead of using the path embedded in the scores file.",
    )
    parser.add_argument("--k_values", type=str, default="1,3,5,10,20,50,100")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--max_issues", type=int, default=0)
    parser.add_argument("--filter_top_n", type=int, default=500)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--same_type_only",
        action="store_true",
        help="Deprecated alias; same-type-only is now the default behavior.",
    )
    parser.add_argument(
        "--all_node_types",
        action="store_true",
        help="Include neighbors of any node type in BFS expansion (disables same-type-only filtering).",
    )
    args = parser.parse_args()

    same_type_only = not args.all_node_types
    if args.filter_top_n < 0:
        raise ValueError("filter_top_n must be >= 0")

    if args.depth <= 0:
        raise ValueError("depth must be > 0")

    k_values = parse_k_values(args.k_values)
    scores = load_issue_scores(args.scores_path)
    issues = scores.get("issues", {})
    if not issues:
        raise ValueError("No issues found in scores file")

    issue_items = list(issues.items())
    end_idx = len(issue_items) if args.end_idx < 0 else min(args.end_idx, len(issue_items))
    start_idx = max(0, args.start_idx)
    selected = issue_items[start_idx:end_idx]
    if args.max_issues > 0:
        selected = selected[: args.max_issues]

    if not selected:
        raise ValueError("No issues selected after slicing")

    processed = 0
    skipped = 0

    issue_iter = tqdm(
        selected,
        desc="Issues",
        leave=True,
        disable=args.no_progress,
    )
    for instance_id, issue_data in issue_iter:
        result = compute_bfs_fields_for_issue(
            issue_data=issue_data,
            k_values=k_values,
            depth=args.depth,
            max_workers=args.max_workers,
            same_type_only=same_type_only,
            filter_top_n=args.filter_top_n,
            show_progress=not args.no_progress,
            issue_id=instance_id,
            graphs_dir=args.graphs_dir,
        )
        issues[instance_id] = result["updated_issue"]
        if result.get("skipped"):
            skipped += 1
        else:
            processed += 1

    scores["issues"] = issues
    scores["bfs_config"] = {
        "depth": args.depth,
        "k_values": k_values,
        "max_workers": args.max_workers,
        "same_type_only": same_type_only,
        "all_node_types": args.all_node_types,
        "filter_top_n": args.filter_top_n,
        "selected_issue_count": len(selected),
        "processed": processed,
        "skipped": skipped,
    }

    save_issue_scores(scores, args.output_path)
    print(
        f"Saved BFS-augmented scores to {args.output_path}. "
        f"selected={len(selected)} processed={processed} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
