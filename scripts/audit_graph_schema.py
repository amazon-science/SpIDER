#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit every graph in the SpIDER-Bench pickle tree to derive the exact schema.

Run before build_hf_dataset.py. Sampling is not enough here: attribute sets vary
by language and rare keys hide in a handful of instances (`is_prototype_method`
appears on 16 nodes of two angular instances). This walks all of them and reports
the complete vocabulary, so the parquet columns are chosen from facts.

    python scripts/audit_graph_schema.py --src /path/to/SpIDER-Bench --out audit.json
"""

import argparse
import collections
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_bench_schema import GROUPS  # noqa: E402


def audit_one(path):
    with open(path, "rb") as fh:
        graph = pickle.load(fh)
    node_keys, edge_keys = collections.Counter(), collections.Counter()
    node_vtypes = collections.defaultdict(collections.Counter)
    edge_vtypes = collections.defaultdict(collections.Counter)
    node_orders, edge_orders = collections.Counter(), collections.Counter()
    import_keytuples = collections.Counter()
    import_vtypes = collections.defaultdict(collections.Counter)
    node_id_types, edge_key_types = collections.Counter(), collections.Counter()
    line_min, line_max = None, None

    for nid, attrs in graph.nodes(data=True):
        node_id_types[type(nid).__name__] += 1
        node_orders[",".join(attrs.keys())] += 1
        for k, v in attrs.items():
            node_keys[k] += 1
            node_vtypes[k][type(v).__name__] += 1
            if k in ("start_line", "end_line") and isinstance(v, int):
                line_min = v if line_min is None else min(line_min, v)
                line_max = v if line_max is None else max(line_max, v)
            if k == "imports" and isinstance(v, list):
                for el in v:
                    if isinstance(el, dict):
                        import_keytuples[",".join(el.keys())] += 1
                        for ik, iv in el.items():
                            import_vtypes[ik][type(iv).__name__] += 1
                    else:
                        import_keytuples[f"<{type(el).__name__}>"] += 1

    for u, v, k, attrs in graph.edges(keys=True, data=True):
        edge_key_types[type(k).__name__] += 1
        edge_orders[",".join(attrs.keys())] += 1
        for kk, vv in attrs.items():
            edge_keys[kk] += 1
            edge_vtypes[kk][type(vv).__name__] += 1

    return {
        "path": str(path),
        "graph_class": type(graph).__name__,
        "graph_attrs": sorted(graph.graph.keys()),
        "num_nodes": graph.number_of_nodes(), "num_edges": graph.number_of_edges(),
        "node_keys": dict(node_keys), "edge_keys": dict(edge_keys),
        "node_vtypes": {k: dict(v) for k, v in node_vtypes.items()},
        "edge_vtypes": {k: dict(v) for k, v in edge_vtypes.items()},
        "node_orders": dict(node_orders), "edge_orders": dict(edge_orders),
        "import_keytuples": dict(import_keytuples),
        "import_vtypes": {k: dict(v) for k, v in import_vtypes.items()},
        "node_id_types": dict(node_id_types), "edge_key_types": dict(edge_key_types),
        "line_min": line_min, "line_max": line_max,
    }


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _merge(dst.setdefault(k, {}), v)
        else:
            dst[k] = dst.get(k, 0) + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    tasks = []
    for benchmark, language, subdir in GROUPS:
        d = Path(args.src) / subdir
        for p in sorted(d.glob("*.pkl")):
            tasks.append((f"{benchmark}/{language}", p))
    print(f"[audit] {len(tasks)} graphs, {args.workers} workers", flush=True)

    per_group = collections.defaultdict(lambda: collections.defaultdict(dict))
    graph_classes, graph_attr_keys = collections.Counter(), collections.Counter()
    lines = [None, None]
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(audit_one, p): g for g, p in tasks}
        for fut in as_completed(futs):
            g = futs[fut]
            r = fut.result()
            graph_classes[r["graph_class"]] += 1
            for k in r["graph_attrs"]:
                graph_attr_keys[k] += 1
            if r["line_min"] is not None:
                lines[0] = r["line_min"] if lines[0] is None else min(lines[0], r["line_min"])
                lines[1] = r["line_max"] if lines[1] is None else max(lines[1], r["line_max"])
            for field in ("node_keys", "edge_keys", "node_vtypes", "edge_vtypes",
                          "node_orders", "edge_orders", "import_keytuples",
                          "import_vtypes", "node_id_types", "edge_key_types"):
                _merge(per_group[g].setdefault(field, {}), r[field])
            done += 1
            if done % 250 == 0:
                print(f"[audit] {done}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)

    out = {
        "num_graphs": len(tasks),
        "graph_classes": dict(graph_classes),
        "graph_attr_keys": dict(graph_attr_keys),
        "line_range": lines,
        "per_group": {g: dict(v) for g, v in per_group.items()},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\n[audit] graph classes: {dict(graph_classes)}")
    print(f"[audit] graph-level attrs: {dict(graph_attr_keys) or 'none'}")
    print(f"[audit] line range: {lines}")
    allnk, allek = collections.Counter(), collections.Counter()
    for g, v in sorted(per_group.items()):
        allnk.update(v["node_keys"]); allek.update(v["edge_keys"])
        print(f"\n### {g}")
        print(f"  node keys: {sorted(v['node_keys'])}")
        print(f"  edge keys: {sorted(v['edge_keys'])}")
        print(f"  node value types: { {k: sorted(t) for k, t in v['node_vtypes'].items()} }")
        print(f"  edge value types: { {k: sorted(t) for k, t in v['edge_vtypes'].items()} }")
        print(f"  node id types: {sorted(v['node_id_types'])}  edge key types: {sorted(v['edge_key_types'])}")
        print(f"  import element shapes: {sorted(v['import_keytuples'])}")
        print(f"  distinct node attr orders: {len(v['node_orders'])}, edge: {len(v['edge_orders'])}")
    print(f"\n[audit] UNION node keys: {sorted(allnk)}")
    print(f"[audit] UNION edge keys: {sorted(allek)}")
    print(f"[audit] -> {args.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
