#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the SpIDER-Bench pickle tree into a Hugging Face-native parquet dataset.

This is the producer side, run once to build the release artifact. Users never
run it -- they run `scripts/materialize_graphs.py`, which is its exact inverse.

    python scripts/build_hf_dataset.py \
        --src /path/to/SpIDER-Bench \
        --out /path/to/SpIDER-Bench-HF

The source tree is only ever read. Every graph is split into `nodes` and `edges`
rows (see scripts/spider_bench_schema.py for why the schema looks the way it
does), sharded so that no instance is ever split across a file and row-group
boundaries fall on instance boundaries -- which is what lets a reader pull one
instance without scanning a whole shard.

Two stray artifacts in the source tree are excluded by construction, because only
`*.pkl` directly inside a group directory is picked up:
  SWE-PolyBench/python/processed/{pre_filter,pre_transform}.pt   (PyG scratch)
  SWE-PolyBench/typescript/SWE-PolyBench/graph_index_typescript_v1.0/...
      a nested stray holding a degenerate 1-node graph, unreachable by the
      runners (which glob the flat language directory) and absent from every
      released result.
"""

import argparse
import collections
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_bench_schema import (  # noqa: E402
    EDGE_SCHEMA, GROUPS, NODE_SCHEMA, graph_to_tables, group_rel_dir,
)

_TRAILING_NUMS = re.compile(r"(-\d+)+$")


def repo_of(instance_id):
    """`anuraghazra__github-readme-stats-58-51` -> `anuraghazra/github-readme-stats`."""
    if "__" not in instance_id:
        return None
    owner, rest = instance_id.split("__", 1)
    return f"{owner}/{_TRAILING_NUMS.sub('', rest)}"


def plan_shards(src, shard_src_bytes, limit=None):
    """Contiguous instance runs, cut on cumulative source size.

    Output size tracks source size closely (~10%), so cutting on source bytes
    gives evenly sized parquet shards without a pre-pass.
    """
    jobs = []
    for benchmark, language, subdir in GROUPS:
        d = Path(src) / subdir
        if not d.is_dir():
            print(f"[plan] WARN: missing {d}, skipping", file=sys.stderr)
            continue
        files = sorted(p for p in d.glob("*.pkl"))  # flat only: excludes strays
        if limit:
            files = files[:limit]
        if not files:
            continue
        chunks, cur, cur_bytes = [], [], 0
        for f in files:
            cur.append(f)
            cur_bytes += f.stat().st_size
            if cur_bytes >= shard_src_bytes:
                chunks.append(cur)
                cur, cur_bytes = [], 0
        if cur:
            chunks.append(cur)
        for i, chunk in enumerate(chunks):
            jobs.append({
                "benchmark": benchmark, "language": language,
                "shard_idx": i, "n_shards": len(chunks),
                "files": [str(p) for p in chunk],
            })
    return jobs


def _shard_paths(out, benchmark, language, shard_idx, n_shards):
    rel = group_rel_dir(benchmark, language)
    n = f"data/nodes/{rel}/nodes-{shard_idx:05d}-of-{n_shards:05d}.parquet"
    e = f"data/edges/{rel}/edges-{shard_idx:05d}-of-{n_shards:05d}.parquet"
    return n, e


class _RowGroupBuffer:
    """Buffers whole instances, flushing a row group once it is big enough.

    Flushing only ever happens on an instance boundary, so a row group never
    splits an instance and a reader can seek to one instance by row-group stats.
    """

    def __init__(self, writer, target_bytes):
        self.writer, self.target = writer, target_bytes
        self.tables, self.nbytes = [], 0

    def add(self, table):
        if table.num_rows == 0:
            return
        self.tables.append(table)
        self.nbytes += table.nbytes
        if self.nbytes >= self.target:
            self.flush()

    def flush(self):
        if not self.tables:
            return
        t = pa.concat_tables(self.tables)
        self.writer.write_table(t, row_group_size=t.num_rows)
        self.tables, self.nbytes = [], 0


def convert_shard(job, src, out, cache_dir, row_group_bytes, compression, level, resume):
    benchmark, language = job["benchmark"], job["language"]
    nrel, erel = _shard_paths(out, benchmark, language, job["shard_idx"], job["n_shards"])
    npath, epath = Path(out) / nrel, Path(out) / erel
    npath.parent.mkdir(parents=True, exist_ok=True)
    epath.parent.mkdir(parents=True, exist_ok=True)

    stats_path = Path(cache_dir) / (nrel.replace("/", "__") + ".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and npath.exists() and epath.exists() and stats_path.exists():
        return json.loads(stats_path.read_text())

    ntmp, etmp = npath.with_suffix(".tmp"), epath.with_suffix(".tmp")
    stats = []
    opts = dict(compression=compression, compression_level=level, use_dictionary=True)
    with pq.ParquetWriter(ntmp, NODE_SCHEMA, **opts) as nw, \
         pq.ParquetWriter(etmp, EDGE_SCHEMA, **opts) as ew:
        nbuf, ebuf = _RowGroupBuffer(nw, row_group_bytes), _RowGroupBuffer(ew, row_group_bytes)
        for f in job["files"]:
            instance_id = Path(f).name[:-len(".pkl")]
            with open(f, "rb") as fh:
                graph = pickle.load(fh)
            ncols, ecols = graph_to_tables(instance_id, graph)
            tn = pa.table({k: ncols[k] for k in NODE_SCHEMA.names}, schema=NODE_SCHEMA)
            te = pa.table({k: ecols[k] for k in EDGE_SCHEMA.names}, schema=EDGE_SCHEMA)
            nbuf.add(tn)
            ebuf.add(te)

            ntypes = collections.Counter(t for t in ncols["type"] if t is not None)
            etypes = collections.Counter(t for t in ecols["type"] if t is not None)
            stats.append({
                "instance_id": instance_id, "benchmark": benchmark, "language": language,
                "repo": repo_of(instance_id),
                "num_nodes": graph.number_of_nodes(), "num_edges": graph.number_of_edges(),
                "node_types": dict(ntypes), "edge_types": dict(etypes),
                "graph_class": type(graph).__name__,
                "graph_attrs": json.dumps(graph.graph, sort_keys=True) if graph.graph else None,
                "nodes_shard": nrel, "edges_shard": erel,
                "node_extras": sum(1 for x in ncols["extra_attrs"] if x),
                "edge_extras": sum(1 for x in ecols["extra_attrs"] if x),
                "source_bytes": os.path.getsize(f),
            })
            del graph, ncols, ecols, tn, te
        nbuf.flush()
        ebuf.flush()
    os.replace(ntmp, npath)
    os.replace(etmp, epath)
    stats_path.write_text(json.dumps(stats))
    return stats


def _worker(args):
    return convert_shard(*args)


def build_instances_table(all_stats, out):
    """The small landing table: one row per instance, viewer's default config."""
    node_types = sorted({t for s in all_stats for t in s["node_types"]})
    edge_types = sorted({t for s in all_stats for t in s["edge_types"]})
    all_stats = sorted(all_stats, key=lambda s: (s["benchmark"], s["language"], s["instance_id"]))

    cols = collections.OrderedDict()
    for k in ("instance_id", "benchmark", "language", "repo"):
        cols[k] = [s[k] for s in all_stats]
    cols["num_nodes"] = [s["num_nodes"] for s in all_stats]
    cols["num_edges"] = [s["num_edges"] for s in all_stats]
    for t in node_types:
        cols[f"n_nodes_{t}"] = [s["node_types"].get(t, 0) for s in all_stats]
    for t in edge_types:
        cols[f"n_edges_{t}"] = [s["edge_types"].get(t, 0) for s in all_stats]
    for k in ("nodes_shard", "edges_shard", "graph_attrs"):
        cols[k] = [s[k] for s in all_stats]

    fields = [
        pa.field("instance_id", pa.string()), pa.field("benchmark", pa.string()),
        pa.field("language", pa.string()), pa.field("repo", pa.string()),
        pa.field("num_nodes", pa.int32()), pa.field("num_edges", pa.int32()),
    ]
    fields += [pa.field(f"n_nodes_{t}", pa.int32()) for t in node_types]
    fields += [pa.field(f"n_edges_{t}", pa.int32()) for t in edge_types]
    fields += [pa.field("nodes_shard", pa.string()), pa.field("edges_shard", pa.string()),
               pa.field("graph_attrs", pa.string())]
    schema = pa.schema(fields)

    table = pa.table(cols, schema=schema)
    path = Path(out) / "data/instances/instances-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=9)
    return table, node_types, edge_types


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="read-only SpIDER-Bench pickle tree")
    ap.add_argument("--out", required=True, help="output directory for the parquet dataset")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-src-bytes", type=float, default=2.5e9,
                    help="cut a shard once this many source bytes accumulate (~10%% of "
                         "this lands in the parquet file)")
    ap.add_argument("--row-group-bytes", type=float, default=128e6)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--compression-level", type=int, default=9)
    ap.add_argument("--limit", type=int, default=None, help="first N instances per group (smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip shards already written")
    ap.add_argument("--cache-dir", default=None,
                    help="resume bookkeeping; kept outside --out so the dataset tree stays "
                         "clean (default: <out>-buildcache)")
    ap.add_argument("--report", default=None, help="where to write the conversion report json")
    args = ap.parse_args()

    src, out = os.path.abspath(args.src), os.path.abspath(args.out)
    if os.path.commonpath([src, out]) == src:
        sys.exit(f"refusing to write inside the source tree: {out}")

    jobs = plan_shards(src, args.shard_src_bytes, args.limit)
    total_files = sum(len(j["files"]) for j in jobs)
    print(f"[build] {len(jobs)} shards, {total_files} instances, {args.workers} workers")
    for j in jobs:
        gb = sum(os.path.getsize(f) for f in j["files"]) / 1e9
        print(f"[plan]   {j['benchmark']}/{j['language']} shard {j['shard_idx']+1}/{j['n_shards']}"
              f"  {len(j['files'])} instances  {gb:.2f} GB source")

    t0 = time.time()
    all_stats, done = [], 0
    cache_dir = args.cache_dir or f"{out.rstrip('/')}-buildcache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[build] build cache (kept out of the dataset tree) -> {cache_dir}")
    payload = [(j, src, out, cache_dir, int(args.row_group_bytes), args.compression,
                args.compression_level, args.resume) for j in jobs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, p): p[0] for p in payload}
        for fut in as_completed(futs):
            j = futs[fut]
            stats = fut.result()
            all_stats.extend(stats)
            done += 1
            print(f"[build] {done}/{len(jobs)} done: {j['benchmark']}/{j['language']} "
                  f"shard {j['shard_idx']+1}/{j['n_shards']} ({len(stats)} instances) "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    table, node_types, edge_types = build_instances_table(all_stats, out)

    # Carry the repo-level files over verbatim; the dataset card is written
    # separately by scripts/write_dataset_card.py from the report below.
    import shutil
    for name in ("LICENSE", "notice.md", ".gitattributes"):
        srcf = os.path.join(src, name)
        if os.path.exists(srcf):
            shutil.copy2(srcf, os.path.join(out, name))
            print(f"[build] copied {name}")

    stray = [str(p) for p in Path(out).rglob("*")
             if p.is_file() and p.suffix != ".parquet"
             and p.name not in ("README.md", "LICENSE", "notice.md", ".gitattributes")]
    if stray:
        sys.exit(f"refusing to finish: non-dataset files in {out}: {stray[:5]}")

    parquet_bytes = sum(p.stat().st_size for p in Path(out).rglob("*.parquet"))
    source_bytes = sum(s["source_bytes"] for s in all_stats)
    report = {
        "num_instances": len(all_stats),
        "num_shards": len(jobs),
        "node_types": node_types,
        "edge_types": edge_types,
        "source_bytes": source_bytes,
        "parquet_bytes": parquet_bytes,
        "ratio": parquet_bytes / source_bytes if source_bytes else None,
        "total_nodes": int(sum(s["num_nodes"] for s in all_stats)),
        "total_edges": int(sum(s["num_edges"] for s in all_stats)),
        "instances_with_extra_node_attrs": sum(1 for s in all_stats if s["node_extras"]),
        "instances_with_extra_edge_attrs": sum(1 for s in all_stats if s["edge_extras"]),
        "instances_with_graph_attrs": sum(1 for s in all_stats if s["graph_attrs"]),
        "graph_classes": sorted({s["graph_class"] for s in all_stats}),
        "per_group": {},
        "elapsed_s": time.time() - t0,
    }
    for s in all_stats:
        g = report["per_group"].setdefault(f"{s['benchmark']}/{s['language']}",
                                           {"instances": 0, "nodes": 0, "edges": 0})
        g["instances"] += 1
        g["nodes"] += s["num_nodes"]
        g["edges"] += s["num_edges"]

    report_path = args.report or os.path.join(os.path.dirname(out), "conversion_report.json")
    with open(report_path, "w") as fh:
        json.dump({"report": report, "instances": all_stats}, fh)

    print(f"\n[build] {len(all_stats)} instances, {report['total_nodes']} nodes, "
          f"{report['total_edges']} edges")
    print(f"[build] {source_bytes/1e9:.1f} GB pickles -> {parquet_bytes/1e9:.2f} GB parquet "
          f"({report['ratio']:.1%})")
    print(f"[build] node types: {node_types}")
    print(f"[build] edge types: {edge_types}")
    print(f"[build] extras: nodes={report['instances_with_extra_node_attrs']} "
          f"edges={report['instances_with_extra_edge_attrs']} "
          f"graph_attrs={report['instances_with_graph_attrs']}")
    print(f"[build] report -> {report_path}")


if __name__ == "__main__":
    main()
