#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schema for the SpIDER-Bench parquet dataset.

The pickle tree stores one `nx.MultiDiGraph` per benchmark instance. The parquet
form splits each graph into two relational tables -- `nodes` and `edges` -- plus a
small `instances` table. Two design points make the round-trip exactly lossless:

`attr_order`
    A comma-joined list of the attribute keys the original dict carried, in the
    original insertion order. It encodes BOTH presence and order, which a plain
    nullable column cannot: parquet null cannot distinguish "attribute absent"
    from "attribute present with value None", and both occur in this data
    (`package` on java nodes, `alias` on edges). Attribute order differs by
    language -- java nodes are (type, code, package, start_line, end_line) while
    python nodes are (type, code, start_line, end_line) -- so a fixed column
    order would not reproduce the original dicts either.

`node_ord` / `edge_ord`
    networkx iteration order is insertion order, and the SpIDER BFS tie-breaks
    depend on it. Rows are rebuilt in this order so traversals reproduce exactly.

`extra_attrs`
    JSON escape hatch for any attribute outside the canonical column set, or one
    whose value has an unexpected type. Nothing can be silently dropped. On
    rebuild it takes precedence over the typed columns.
"""

import pyarrow as pa

# Canonical attribute columns. scripts/audit_graph_schema.py walked all 3,297 graphs
# and these ten node keys / three edge keys are the complete vocabulary.
# Order here is only the column order in the file; per-row attribute order lives
# in `attr_order`.
NODE_ATTR_COLS = [
    "type", "code", "start_line", "end_line",   # every language
    "package",                                  # java
    "imports",                                  # javascript / typescript
    "method_name", "class_name", "parent_type",  # typescript
    "is_prototype_method",                       # typescript
]
EDGE_ATTR_COLS = ["type", "alias", "module"]

# `imports` (javascript / typescript) is a list of dicts, always exactly
# {"type": str, "module": str, "alias": str|None} in that key order. Typing it as
# a struct keeps it queryable in the viewer; any element that deviates from this
# shape falls back to extra_attrs, so an unseen variant cannot corrupt the column.
IMPORT_FIELDS = ("type", "module", "alias")
IMPORT_STRUCT = pa.struct([(f, pa.string()) for f in IMPORT_FIELDS])

NODE_SCHEMA = pa.schema([
    ("instance_id", pa.string()),
    ("node_ord", pa.int32()),
    ("node_id", pa.string()),
    ("type", pa.string()),
    ("code", pa.string()),
    ("start_line", pa.int32()),
    ("end_line", pa.int32()),
    ("package", pa.string()),
    ("imports", pa.list_(IMPORT_STRUCT)),
    ("method_name", pa.string()),
    ("class_name", pa.string()),
    ("parent_type", pa.string()),
    ("is_prototype_method", pa.bool_()),
    ("attr_order", pa.string()),
    ("extra_attrs", pa.string()),
])

EDGE_SCHEMA = pa.schema([
    ("instance_id", pa.string()),
    ("edge_ord", pa.int32()),
    ("src", pa.string()),
    ("dst", pa.string()),
    ("edge_key", pa.int32()),
    ("type", pa.string()),
    ("alias", pa.string()),
    ("module", pa.string()),
    ("attr_order", pa.string()),
    ("extra_attrs", pa.string()),
])

# (benchmark, language, source subdirectory). SWE-bench Verified has no language
# level in the tree; it is all python.
GROUPS = [
    ("SWE-bench_Verified", "python",     "SWE-bench_Verified"),
    ("SWE-PolyBench",      "python",     "SWE-PolyBench/python"),
    ("SWE-PolyBench",      "java",       "SWE-PolyBench/java"),
    ("SWE-PolyBench",      "javascript", "SWE-PolyBench/javascript"),
    ("SWE-PolyBench",      "typescript", "SWE-PolyBench/typescript"),
    ("Multi-SWE-bench",    "java",       "Multi-SWE-bench/java"),
    ("Multi-SWE-bench",    "javascript", "Multi-SWE-bench/javascript"),
    ("Multi-SWE-bench",    "typescript", "Multi-SWE-bench/typescript"),
]

# Split names inside the `nodes` / `edges` configs. They cannot be two splits of
# one config: `datasets` casts every split in a config to the first split's
# schema, and node rows and edge rows have different columns.
SPLIT_NAMES = {
    ("SWE-bench_Verified", "python"):     "swe_bench_verified",
    ("SWE-PolyBench", "python"):          "swe_polybench_python",
    ("SWE-PolyBench", "java"):            "swe_polybench_java",
    ("SWE-PolyBench", "javascript"):      "swe_polybench_javascript",
    ("SWE-PolyBench", "typescript"):      "swe_polybench_typescript",
    ("Multi-SWE-bench", "java"):          "multi_swe_bench_java",
    ("Multi-SWE-bench", "javascript"):    "multi_swe_bench_javascript",
    ("Multi-SWE-bench", "typescript"):    "multi_swe_bench_typescript",
}


def group_rel_dir(benchmark: str, language: str) -> str:
    """Path fragment under data/{nodes,edges}/ for one benchmark+language."""
    if benchmark == "SWE-bench_Verified":
        return benchmark
    return f"{benchmark}/{language}"


# --------------------------------------------------------------------------- #
# graph <-> tables. These two functions are exact inverses; the build script
# asserts it with networkx.utils.graphs_equal over every graph in the tree.
# --------------------------------------------------------------------------- #

import json
import sys

_INT32_MIN, _INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def _fits(key, value):
    """Can `value` go in its typed column, or must it fall back to extra_attrs?

    None always fits (stored as null; presence is recorded by attr_order).
    """
    if value is None:
        return True
    if key in ("type", "code", "package", "method_name", "class_name",
               "parent_type", "alias", "module"):
        return isinstance(value, str)
    if key == "is_prototype_method":
        return isinstance(value, bool)
    if key in ("start_line", "end_line"):
        return (isinstance(value, int) and not isinstance(value, bool)
                and _INT32_MIN <= value <= _INT32_MAX)
    if key == "imports":
        return isinstance(value, list) and all(
            isinstance(x, dict) and tuple(x.keys()) == IMPORT_FIELDS
            and all(v is None or isinstance(v, str) for v in x.values())
            for x in value)
    return False


def _split_attrs(attrs, cols):
    """-> (typed values dict, attr_order string, extra_attrs json or None)."""
    typed = {c: None for c in cols}
    extra = {}
    for k, v in attrs.items():
        if k in typed and _fits(k, v):
            typed[k] = v
        else:
            extra[k] = v
    order = ",".join(attrs.keys())
    return typed, order, (json.dumps(extra, sort_keys=True) if extra else None)


def graph_to_tables(instance_id, graph):
    """One nx.MultiDiGraph -> (nodes dict-of-columns, edges dict-of-columns)."""
    ncols = {name: [] for name in NODE_SCHEMA.names}
    for ord_, (node_id, attrs) in enumerate(graph.nodes(data=True)):
        if not isinstance(node_id, str):
            raise TypeError(f"{instance_id}: non-str node id {node_id!r}")
        typed, order, extra = _split_attrs(attrs, NODE_ATTR_COLS)
        ncols["instance_id"].append(instance_id)
        ncols["node_ord"].append(ord_)
        ncols["node_id"].append(node_id)
        ncols["attr_order"].append(order)
        ncols["extra_attrs"].append(extra)
        for c in NODE_ATTR_COLS:
            ncols[c].append(typed[c])

    ecols = {name: [] for name in EDGE_SCHEMA.names}
    for ord_, (src, dst, key, attrs) in enumerate(graph.edges(keys=True, data=True)):
        if not (isinstance(key, int) and not isinstance(key, bool)
                and _INT32_MIN <= key <= _INT32_MAX):
            raise TypeError(f"{instance_id}: unrepresentable edge key {key!r}")
        typed, order, extra = _split_attrs(attrs, EDGE_ATTR_COLS)
        ecols["instance_id"].append(instance_id)
        ecols["edge_ord"].append(ord_)
        ecols["src"].append(src)
        ecols["dst"].append(dst)
        ecols["edge_key"].append(key)
        ecols["attr_order"].append(order)
        ecols["extra_attrs"].append(extra)
        for c in EDGE_ATTR_COLS:
            ecols[c].append(typed[c])

    return ncols, ecols


def _rebuild_attrs(row_get, ord_, keys, extra_json, interned):
    if not keys:
        return {}
    extra = json.loads(extra_json) if extra_json else {}
    attrs = {}
    for k in keys:
        # extra_attrs wins: it holds the true value whenever the typed column
        # could not represent it.
        attrs[k] = extra[k] if k in extra else interned(k, row_get(k, ord_))
    return attrs


# Values worth pooling on rebuild: low cardinality, and shared objects in the
# original graphs. `code` is excluded -- the strings are large and near-unique,
# so hashing them would cost more than it saves.
_POOLED = frozenset(("type", "package", "method_name", "class_name", "parent_type",
                     "alias", "module"))


def tables_to_graph(node_cols, edge_cols, graph_attrs=None):
    """(nodes, edges) column dicts -> the original nx.MultiDiGraph.

    Rows must be in `node_ord` / `edge_ord` order; the caller sorts. Nodes are
    added before edges so networkx insertion order matches the original.

    String identity is reconstructed, not just string equality. In the source
    graphs one `str` object is shared by a node key and by every edge endpoint
    naming it, and pickle memoizes by identity -- so a naive rebuild, which
    allocates a fresh string per parquet cell, writes each node id out again for
    all 671M edge endpoints and inflates the pickles ~2.5-3.3x. Pooling node ids,
    attribute keys and low-cardinality values restores the original sizes.
    """
    import networkx as nx

    graph = nx.MultiDiGraph()
    if graph_attrs:
        graph.graph.update(graph_attrs)

    pool = {}

    def interned(key, value):
        if key not in _POOLED or value is None:
            return value
        got = pool.get(value)
        if got is None:
            pool[value] = value
            return value
        return got

    key_cache = {}

    def keys_of(order):
        """attr_order -> interned key tuple. Only a handful of distinct orders
        exist per language, so this collapses to a few cached tuples."""
        got = key_cache.get(order)
        if got is None:
            got = tuple(sys.intern(k) for k in order.split(",")) if order else ()
            key_cache[order] = got
        return got

    import_keys = tuple(sys.intern(f) for f in IMPORT_FIELDS)

    canon = {}
    nget = lambda k, i: node_cols[k][i]
    node_ids = node_cols["node_id"]
    for i, node_id in enumerate(node_ids):
        attrs = _rebuild_attrs(nget, i, keys_of(node_cols["attr_order"][i]),
                               node_cols["extra_attrs"][i], interned)
        imports = attrs.get("imports")
        if imports:
            attrs["imports"] = [
                None if el is None else
                {import_keys[j]: interned("module", el[IMPORT_FIELDS[j]])
                 for j in range(len(IMPORT_FIELDS))}
                for el in imports]
        canon[node_id] = node_id
        graph.add_node(node_id, **attrs)

    eget = lambda k, i: edge_cols[k][i]
    srcs, dsts = edge_cols["src"], edge_cols["dst"]
    for i in range(len(srcs)):
        src, dst = srcs[i], dsts[i]
        graph.add_edge(canon.get(src, src), canon.get(dst, dst),
                       key=edge_cols["edge_key"][i],
                       **_rebuild_attrs(eget, i, keys_of(edge_cols["attr_order"][i]),
                                        edge_cols["extra_attrs"][i], interned))
    return graph
