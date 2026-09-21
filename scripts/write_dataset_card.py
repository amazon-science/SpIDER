#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Hub dataset card (README.md) for the parquet SpIDER-Bench release.

Numbers come from the conversion report, so the card cannot drift from the data.
The YAML front matter is what makes the dataset viewer work: `configs:` maps each
benchmark+language to its parquet shards, with `nodes` and `edges` as named
splits, and `instances` as the small default config the viewer opens on.

    python scripts/write_dataset_card.py --out /path/to/SpIDER-Bench-HF \
        --report /path/to/conversion_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_bench_schema import GROUPS, SPLIT_NAMES, group_rel_dir  # noqa: E402

CITATION = """@article{chaudhari2024spider,
    title={SpIDER: Spatially Informed Dense Embedding Retrieval for Software Issue Localization},
    author={Chaudhari, Shravan and Jacob, Rahul Thomas and Goswami, Mononito and Cao, Jiajun and Rashid, Shihab and Bock, Christian},
    journal={arXiv preprint arXiv:2512.16956},
    year={2024}
}"""


def yaml_front_matter(report):
    lines = ["---",
             "license: other",
             "license_name: other",
             "license_link: LICENSE",
             "task_categories:",
             "- feature-extraction",
             "tags:",
             "- code",
             "- software-engineering",
             "- issue-localization",
             "- code-graph",
             "- swe-bench",
             "pretty_name: SpIDER-Bench",
             "configs:",
             "- config_name: instances",
             "  default: true",
             "  data_files:",
             "  - split: train",
             "    path: data/instances/*.parquet"]
    for kind in ("nodes", "edges"):
        lines += [f"- config_name: {kind}", "  data_files:"]
        for benchmark, language, _ in GROUPS:
            key = f"{benchmark}/{language}"
            if key not in report["per_group"]:
                continue
            rel = group_rel_dir(benchmark, language)
            lines += [f"  - split: {SPLIT_NAMES[(benchmark, language)]}",
                      f"    path: data/{kind}/{rel}/*.parquet"]
    lines.append("---")
    return "\n".join(lines)


def body(report):
    g = report["per_group"]
    rows = []
    for benchmark, language, _ in GROUPS:
        key = f"{benchmark}/{language}"
        if key not in g:
            continue
        s = g[key]
        rows.append(f"| `{SPLIT_NAMES[(benchmark, language)]}` | {benchmark} | {language} | "
                    f"{s['instances']:,} | {s['nodes']:,} | {s['edges']:,} |")
    table = "\n".join(rows)

    node_types = ", ".join(f"`{t}`" for t in report["node_types"])
    edge_types = ", ".join(f"`{t}`" for t in report["edge_types"])

    return f"""
# SpIDER-Bench

Repository dependency graphs for software issue localization — the graph data behind
**SpIDER: Spatially Informed Dense Embedding Retrieval for Software Issue Localization**
([arXiv:2512.16956](https://arxiv.org/abs/2512.16956)).

Each benchmark instance gets one directed multigraph of its repository at the commit the
issue was filed against. Nodes are directories, files, classes and functions carrying
their source; edges are `contains` / `imports` / `inherits` / `invokes` relations between
them. SpIDER uses these graphs to expand a dense-retrieval ranking along the code's own
structure.

**{report['num_instances']:,} graphs · {report['total_nodes']:,} nodes · {report['total_edges']:,} edges**
across three benchmarks and four languages.

## Configurations

Three configs: **`instances`** (default, one row per graph), **`nodes`** and **`edges`**.
`nodes` and `edges` each carry the same eight splits, one per benchmark+language:

| split | benchmark | language | instances | nodes | edges |
|---|---|---|---:|---:|---:|
{table}

`nodes` and `edges` are separate configs rather than two splits of one, because
`datasets` casts every split in a config to a single schema and node rows and edge rows
have different columns.

## Loading

```python
from datasets import load_dataset

# the summary table: one row per instance
inst = load_dataset("AmazonScience/SpIDER-Bench", "instances", split="train")

# the graph tables for one benchmark+language
nodes = load_dataset("AmazonScience/SpIDER-Bench", "nodes", split="swe_polybench_python")
edges = load_dataset("AmazonScience/SpIDER-Bench", "edges", split="swe_polybench_python")
```

Graphs are stored relationally rather than one-row-per-graph because a single instance
reaches hundreds of MB — too large for a parquet row or the viewer.

## Rebuilding the graphs

The reference implementation consumes `networkx.MultiDiGraph` pickles. The SpIDER code
release ships `scripts/materialize_graphs.py`, which rebuilds them and downloads only the
shards it needs:

```bash
python scripts/materialize_graphs.py --out data/SpIDER-Bench
python scripts/materialize_graphs.py --out data/SpIDER-Bench --subset SWE-PolyBench/python
```

To rebuild one graph directly:

```python
import json, networkx as nx, pyarrow.parquet as pq

NODE_COLS = ["type", "code", "start_line", "end_line", "package", "imports",
             "method_name", "class_name", "parent_type", "is_prototype_method"]
EDGE_COLS = ["type", "alias", "module"]

def rebuild(node_rows, edge_rows):
    g = nx.MultiDiGraph()
    for r in sorted(node_rows, key=lambda r: r["node_ord"]):
        g.add_node(r["node_id"], **attrs(r, NODE_COLS))
    for r in sorted(edge_rows, key=lambda r: r["edge_ord"]):
        g.add_edge(r["src"], r["dst"], key=r["edge_key"], **attrs(r, EDGE_COLS))
    return g

def attrs(row, cols):
    if not row["attr_order"]:
        return {{}}
    extra = json.loads(row["extra_attrs"]) if row["extra_attrs"] else {{}}
    return {{k: extra[k] if k in extra else row[k] for k in row["attr_order"].split(",")}}
```

## Schema

### `instances`

| column | type | meaning |
|---|---|---|
| `instance_id` | string | benchmark instance id, globally unique across configs |
| `benchmark` | string | `SWE-bench_Verified`, `SWE-PolyBench`, `Multi-SWE-bench` |
| `language` | string | `python`, `java`, `javascript`, `typescript` |
| `repo` | string | `owner/name` |
| `num_nodes`, `num_edges` | int32 | graph size |
| `n_nodes_<type>`, `n_edges_<type>` | int32 | per-type counts |
| `nodes_shard`, `edges_shard` | string | parquet files holding this instance |
| `graph_attrs` | string | graph-level attributes as JSON, null when none |

### `nodes`

| column | type | meaning |
|---|---|---|
| `instance_id` | string | joins to `instances` |
| `node_ord` | int32 | insertion order — **rebuild in this order** |
| `node_id` | string | `path/to/file.py`, `…:Class`, `…:Class.method` |
| `type` | string | {node_types} |
| `code` | string | source text of the node |
| `start_line`, `end_line` | int32 | 1-based line span in the file |
| `package` | string | java only |
| `imports` | list&lt;struct&lt;type, module, alias&gt;&gt; | javascript / typescript only |
| `method_name`, `class_name`, `parent_type` | string | typescript only |
| `is_prototype_method` | bool | typescript only |
| `attr_order` | string | comma-joined original attribute keys, in order |
| `extra_attrs` | string | JSON for anything outside the typed columns, null when none |

### `edges`

| column | type | meaning |
|---|---|---|
| `instance_id` | string | joins to `instances` |
| `edge_ord` | int32 | insertion order — **rebuild in this order** |
| `src`, `dst` | string | node ids |
| `edge_key` | int32 | parallel-edge key (`MultiDiGraph`) |
| `type` | string | {edge_types} |
| `alias` | string | import alias, where one applies |
| `module` | string | imported module, javascript / typescript |
| `attr_order`, `extra_attrs` | string | as above |

### Why `attr_order` and `*_ord`

Attribute sets differ by language and, within a language, between node kinds — a java node
carries `package`, a directory node carries only `type`. Parquet null cannot distinguish
*attribute absent* from *attribute present with value `None`*, and both occur here. So
`attr_order` records exactly which keys the original dict held and in what order, and it is
what a faithful rebuild iterates. `node_ord` / `edge_ord` preserve networkx insertion order,
which SpIDER's BFS tie-breaks depend on.

Round-tripping every graph in this release through
`networkx.utils.graphs_equal` against the original pickles passes for all
{report['num_instances']:,}, including node order, edge order and per-node attribute key order.

## Citation

```bibtex
{CITATION}
```

## License

See `LICENSE` and `notice.md`. This repository contains code segments under multiple
licenses (MIT, Apache 2.0, BSD, GPL and others) and is adapted from the listed open-source
projects; your use must comply with the relevant segments' licenses.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    report = json.load(open(args.report))["report"]
    text = yaml_front_matter(report) + "\n" + body(report)
    path = Path(args.out) / "README.md"
    path.write_text(text)
    print(f"[card] -> {path} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
