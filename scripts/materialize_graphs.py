#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the SpIDER-Bench repository graphs from the parquet dataset.

SpIDER-Bench ships as parquet so it loads with `datasets` and renders in the Hub
viewer. The pipeline wants one `nx.MultiDiGraph` pickle per instance, which is
what this produces -- byte-for-byte the layout every script here expects:

    <out>/SWE-bench_Verified/{instance_id}.pkl
    <out>/SWE-PolyBench/{python,java,javascript,typescript}/{instance_id}.pkl
    <out>/Multi-SWE-bench/{java,javascript,typescript}/{instance_id}.pkl

Usage:
    # everything, straight from the Hub
    python scripts/materialize_graphs.py --out data/SpIDER-Bench

    # just what you need -- only the shards holding it are downloaded
    python scripts/materialize_graphs.py --out data/SpIDER-Bench --subset SWE-PolyBench/python
    python scripts/materialize_graphs.py --out data/SpIDER-Bench \
        --instance-ids astropy__astropy-12907 django__django-11099

    # from an already-downloaded parquet tree
    python scripts/materialize_graphs.py --out data/SpIDER-Bench --parquet-dir /path/to/SpIDER-Bench-HF

It is resumable: instances whose pickle already exists are skipped unless
--overwrite. `--verify-against <original tree>` re-checks every rebuilt graph
against the original pickles with networkx.utils.graphs_equal.
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_bench_schema import GROUPS, group_rel_dir, tables_to_graph  # noqa: E402

REPO_ID = "AmazonScience/SpIDER-Bench"
INSTANCES_GLOB = "data/instances"


class _Peekable:
    """Iterator with one-item lookahead, so a missing edge group cannot desync."""

    def __init__(self, it):
        self._it, self._peeked, self._has = it, None, False

    def peek(self):
        if not self._has:
            self._peeked = next(self._it, None)
            self._has = True
        return self._peeked

    def take(self):
        v = self.peek()
        self._has = False
        return v


def iter_instances(path):
    """Stream (instance_id, column dict) from one parquet shard.

    Rows are grouped by instance and contiguous, and row groups never split an
    instance, so this holds at most one instance in memory at a time.
    """
    pf = pq.ParquetFile(path)
    buf, cur = [], None
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        ids = table.column("instance_id").to_pylist()
        start = 0
        for i in range(1, len(ids) + 1):
            if i == len(ids) or ids[i] != ids[start]:
                chunk, iid = table.slice(start, i - start), ids[start]
                if cur is not None and iid != cur:
                    yield cur, _to_cols(buf)
                    buf = []
                cur = iid
                buf.append(chunk)
                start = i
    if cur is not None:
        yield cur, _to_cols(buf)


def _to_cols(tables):
    import pyarrow as pa
    return (tables[0] if len(tables) == 1 else pa.concat_tables(tables)).to_pydict()


def _empty_like(path):
    return {name: [] for name in pq.ParquetFile(path).schema_arrow.names}


def materialize_shard(nodes_path, edges_path, out_root, wanted, graph_attrs,
                      dest_of, overwrite, verify_against):
    """Rebuild every wanted instance in one shard pair. Returns per-instance results."""
    import json
    import networkx as nx

    results = []
    edges = _Peekable(iter_instances(edges_path))
    no_edges = _empty_like(edges_path)
    for iid, ncols in iter_instances(nodes_path):
        ecols = no_edges
        nxt = edges.peek()
        if nxt is not None and nxt[0] == iid:
            ecols = edges.take()[1]
        if iid not in wanted:
            continue

        dest = Path(out_root) / dest_of[iid] / f"{iid}.pkl"
        if dest.exists() and not overwrite and not verify_against:
            results.append({"instance_id": iid, "status": "skipped"})
            continue

        attrs = graph_attrs.get(iid)
        graph = tables_to_graph(ncols, ecols, json.loads(attrs) if attrs else None)

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dest)

        rec = {"instance_id": iid, "status": "written",
               "num_nodes": graph.number_of_nodes(), "num_edges": graph.number_of_edges()}
        if verify_against:
            orig_path = Path(verify_against) / dest_of[iid] / f"{iid}.pkl"
            with open(orig_path, "rb") as fh:
                original = pickle.load(fh)
            rec["graphs_equal"] = bool(nx.utils.graphs_equal(original, graph))
            rec["node_order_equal"] = list(original.nodes()) == list(graph.nodes())
            rec["edge_order_equal"] = (list(original.edges(keys=True))
                                       == list(graph.edges(keys=True)))
            rec["attr_order_equal"] = all(
                list(original.nodes[n].keys()) == list(graph.nodes[n].keys())
                for n in original.nodes())
        results.append(rec)
    return results


def _worker(a):
    return materialize_shard(*a)


def resolve_instances(args):
    """-> (instances arrow table filtered to what was asked for, parquet root resolver)."""
    if args.parquet_dir:
        root = Path(args.parquet_dir)
        cands = sorted((root / INSTANCES_GLOB).glob("*.parquet"))
        if not cands:
            sys.exit(f"no instances parquet under {root/INSTANCES_GLOB}")
        table = pq.read_table(cands[0])
        fetch = lambda rel: str(root / rel)
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(args.repo_id, repo_type="dataset", token=args.token)
        inst = sorted(f for f in files if f.startswith(INSTANCES_GLOB) and f.endswith(".parquet"))
        if not inst:
            sys.exit(f"{args.repo_id} has no {INSTANCES_GLOB}/*.parquet -- is it the "
                     f"parquet release? For the legacy pickle tree use download_bench.py.")
        table = pq.read_table(hf_hub_download(args.repo_id, inst[0], repo_type="dataset",
                                              token=args.token))
        fetch = lambda rel: hf_hub_download(args.repo_id, rel, repo_type="dataset",
                                            token=args.token)
    return table, fetch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/SpIDER-Bench", help="where the pickles go")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--parquet-dir", default=None,
                    help="use an already-downloaded parquet tree instead of the Hub")
    ap.add_argument("--subset", action="append", default=None,
                    help="'SWE-PolyBench/python' or 'SWE-PolyBench'. Repeatable.")
    ap.add_argument("--instance-ids", nargs="*", default=None)
    ap.add_argument("--instance-ids-file", default=None, help="one instance id per line")
    ap.add_argument("--limit", type=int, default=None, help="first N instances (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-against", default=None,
                    help="original pickle tree; assert graphs_equal for every instance")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    table, fetch = resolve_instances(args)
    cols = table.to_pydict()
    rows = list(zip(cols["instance_id"], cols["benchmark"], cols["language"],
                    cols["nodes_shard"], cols["edges_shard"], cols["graph_attrs"]))

    if args.subset:
        keep = set()
        for s in args.subset:
            s = s.strip("/")
            for benchmark, language, _ in GROUPS:
                if s in (benchmark, f"{benchmark}/{language}", group_rel_dir(benchmark, language)):
                    keep.add((benchmark, language))
        if not keep:
            sys.exit(f"--subset {args.subset} matched no group")
        rows = [r for r in rows if (r[1], r[2]) in keep]
    ids = set(args.instance_ids or [])
    if args.instance_ids_file:
        ids |= {l.strip() for l in open(args.instance_ids_file) if l.strip()}
    if ids:
        rows = [r for r in rows if r[0] in ids]
        missing = ids - {r[0] for r in rows}
        if missing:
            print(f"[materialize] WARN: {len(missing)} unknown instance ids, e.g. "
                  f"{sorted(missing)[:3]}", file=sys.stderr)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("nothing selected")

    shards = {}
    for iid, benchmark, language, nshard, eshard, gattrs in rows:
        s = shards.setdefault((nshard, eshard), {"wanted": set(), "dest": {}, "attrs": {}})
        s["wanted"].add(iid)
        s["dest"][iid] = group_rel_dir(benchmark, language)
        if gattrs:
            s["attrs"][iid] = gattrs

    print(f"[materialize] {len(rows)} instances from {len(shards)} shard pairs "
          f"-> {args.out}", flush=True)

    t0 = time.time()
    payload = []
    for (nshard, eshard), s in sorted(shards.items()):
        npath, epath = fetch(nshard), fetch(eshard)
        payload.append((npath, epath, args.out, s["wanted"], s["attrs"], s["dest"],
                        args.overwrite, args.verify_against))
    print(f"[materialize] parquet ready [{time.time()-t0:.0f}s], rebuilding", flush=True)

    results, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, p) for p in payload]
        for fut in as_completed(futs):
            results.extend(fut.result())
            done += 1
            print(f"[materialize] {done}/{len(payload)} shards [{time.time()-t0:.0f}s]",
                  flush=True)

    written = [r for r in results if r["status"] == "written"]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"\n[materialize] {len(written)} written, {len(skipped)} already present "
          f"[{time.time()-t0:.0f}s]")

    exit_code = 0
    if args.verify_against:
        checks = ["graphs_equal", "node_order_equal", "edge_order_equal", "attr_order_equal"]
        for c in checks:
            bad = [r["instance_id"] for r in written if not r.get(c)]
            print(f"[verify] {c}: {len(written)-len(bad)}/{len(written)} pass"
                  + (f"  FAILED: {bad[:5]}" if bad else ""))
            if bad:
                exit_code = 1
        print("[verify] " + ("ALL PASS" if exit_code == 0 else "FAILURES -- see above"))

    if args.report:
        import json
        with open(args.report, "w") as fh:
            json.dump(results, fh)
        print(f"[materialize] report -> {args.report}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
