#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch SpIDER-Bench from the Hugging Face Hub and rebuild the repository graphs.

SpIDER-Bench is published as parquet, so it loads with `datasets` and renders in
the Hub viewer. The pipeline wants one `nx.MultiDiGraph` pickle per instance, so
this does both steps: download the parquet, then materialize the pickles into the
layout every script here expects.

    data/SpIDER-Bench/SWE-bench_Verified/{instance_id}.pkl
    data/SpIDER-Bench/SWE-PolyBench/{python,java,javascript,typescript}/{instance_id}.pkl
    data/SpIDER-Bench/Multi-SWE-bench/{java,javascript,typescript}/{instance_id}.pkl

Usage:
    python scripts/download_bench.py --out data/SpIDER-Bench
    python scripts/download_bench.py --out data/SpIDER-Bench --subset SWE-PolyBench/python
    HF_TOKEN=... python scripts/download_bench.py --out data/SpIDER-Bench

The parquet is ~7 GB; the materialized pickles are ~73 GB, so plan for both unless
you pass --no-keep-parquet. For a subset, `scripts/materialize_graphs.py` on its
own is cheaper -- it pulls only the shards holding what you asked for, instead of
snapshotting the whole repo first.
"""

import argparse
import os
import shutil
import subprocess
import sys

from huggingface_hub import snapshot_download

REPO_ID = "AmazonScience/SpIDER-Bench"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/SpIDER-Bench",
                    help="where the rebuilt graph pickles go")
    ap.add_argument("--parquet-dir", default=None,
                    help="where to put the downloaded parquet (default: <out>-parquet)")
    ap.add_argument("--subset", action="append", default=None,
                    help="restrict to a group, e.g. 'SWE-PolyBench/python'. Repeatable; "
                         "omit to fetch everything.")
    ap.add_argument("--max_workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--workers", type=int, default=8, help="parallel graph rebuilds")
    ap.add_argument("--parquet-only", action="store_true",
                    help="download the parquet and stop, skipping the rebuild")
    ap.add_argument("--no-keep-parquet", action="store_true",
                    help="delete the parquet snapshot once the pickles are built")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    parquet_dir = args.parquet_dir or f"{args.out.rstrip('/')}-parquet"

    # The instances table is tiny and always needed; graph shards are filtered by
    # group so a --subset download stays small.
    allow = ["README.md", "LICENSE", "notice.md", "data/instances/*"]
    if args.subset:
        for s in args.subset:
            s = s.strip("/")
            allow += [f"data/nodes/{s}/*", f"data/edges/{s}/*"]
    else:
        allow = None

    print(f"downloading {REPO_ID} -> {parquet_dir}"
          + (f"  (subsets: {', '.join(args.subset)})" if args.subset else ""))
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=parquet_dir,
        allow_patterns=allow,
        max_workers=args.max_workers,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"parquet -> {parquet_dir}")

    if args.parquet_only:
        return

    cmd = [sys.executable, os.path.join(HERE, "materialize_graphs.py"),
           "--parquet-dir", parquet_dir, "--out", args.out,
           "--workers", str(args.workers)]
    for s in args.subset or []:
        cmd += ["--subset", s]
    if args.overwrite:
        cmd.append("--overwrite")
    print("rebuilding graphs: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)

    if args.no_keep_parquet:
        shutil.rmtree(parquet_dir, ignore_errors=True)
        print(f"removed {parquet_dir}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
