#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recursively summarize SPIDER result JSONs into per-directory tables.

For EVERY directory at or below --root (default: the directory holding this
script), writes two files summarizing all result JSONs at or below it:

    SUMMARY.txt   fixed-width, human-readable
    summary.csv   long-form, one row per (lang, config, gt_view, filter)

What each summary contains
--------------------------
* Metrics are reported SEPARATELY for the two ground-truth views, as two
  independent tables:
    - spider : SWE-PolyBench `modified_nodes` (function leaves) ∩ candidate pool
    - ours   : LocAgent GT (patch-line overlap)
  SWE-bench Verified carries no `modified_nodes` field, so only the `ours` view
  appears for it.
* Each view is reported under two instance filters:
    - all       : every succeeded instance with a non-empty GT for that view
                  (the pipeline's own convention)
    - func_only : the above, restricted to is_func_only instances (patch touches
                  only functions, no class-body edits). is_func_only comes from
                  SWE-PolyBench metadata; see below.
* Each metric row also carries the no-LLM DENSE RETRIEVAL baseline (top-L of the
  ranking) over the same evaluated set, plus recall lift.
* A separate TOKEN section (GT-view independent) reports, per instance and per
  call: input, output, input+output, reasoning, input+output+reasoning.

Token accounting
----------------
`output` here means VISIBLE output tokens = completion_tokens - reasoning_tokens,
so `input+output+reasoning` is the full billed total with no double counting.
Reasoning tokens are read from usage.reasoning_tokens; a run against an endpoint
that does not report them shows reasoning = 0, and `output` then equals the raw
completion_tokens.

Metadata cache
--------------
is_func_only lives in the score pickles, not the result JSONs, so a small cache
(_metadata_cache.json, ~one line per instance) is built once from the pickles in
--pickle-dir. Rebuild with --rebuild-cache. Without the pickles the func_only
tables are simply omitted; the `all` tables are unaffected.

Usage
-----
    python summarize.py --root results/<model>
    python summarize.py --root results/<model> --rebuild-cache
    python summarize.py --root results/<model> --pickle-dir data/node_embeddings/swepolybench
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
# Score pickles carry the is_func_only flag; overridable with --pickle-dir.
DEFAULT_PICKLE_DIR = Path(
    os.environ.get("SCORES_ROOT", str(SCRIPT_DIR / "data" / "node_embeddings"))
) / "swepolybench"
PICKLE_DIR = DEFAULT_PICKLE_DIR
CACHE_FILE = SCRIPT_DIR / "_metadata_cache.json"
LANGS = ["python", "java", "javascript", "typescript"]

# The two GT views: (label, gt field, metric prefix, baseline prefix)
VIEWS = [
    ("spider", "spider_ground_truth", "spider_", "spider_baseline_"),
    ("ours",   "ground_truth",        "",        "baseline_"),
]
FILTERS = ["all", "func_only"]


# --------------------------------------------------------------------------- #
# Metadata cache: {lang: {instance_id: {is_func_only, is_mixed, nf, nc}}}
# --------------------------------------------------------------------------- #
def build_metadata_cache() -> Dict[str, Dict[str, dict]]:
    import pickle
    import gc
    cache: Dict[str, Dict[str, dict]] = {}
    for lang in LANGS:
        pk = PICKLE_DIR / f"test_{lang}_issue_node_scores_bfs4_filtered500.pkl"
        if not pk.exists():
            print(f"  [cache] WARN missing pickle for {lang}: {pk}", file=sys.stderr)
            cache[lang] = {}
            continue
        print(f"  [cache] loading {lang} ({pk.stat().st_size/1e9:.1f}GB) ...", file=sys.stderr)
        with open(pk, "rb") as fh:
            d = pickle.load(fh)
        lang_meta: Dict[str, dict] = {}
        for iid, iss in d["issues"].items():
            f = iss.get("issue_fields", {}) or {}
            lang_meta[iid] = {
                "is_func_only": bool(f.get("is_func_only")),
                "is_mixed": bool(f.get("is_mixed")),
                "num_func_changes": f.get("num_func_changes"),
                "num_class_changes": f.get("num_class_changes"),
            }
        cache[lang] = lang_meta
        del d
        gc.collect()
    CACHE_FILE.write_text(json.dumps(cache))
    print(f"  [cache] wrote {CACHE_FILE} "
          f"({sum(len(v) for v in cache.values())} instances)", file=sys.stderr)
    return cache


def load_metadata_cache(rebuild: bool = False) -> Dict[str, Dict[str, dict]]:
    if rebuild or not CACHE_FILE.exists():
        return build_metadata_cache()
    return json.loads(CACHE_FILE.read_text())


# --------------------------------------------------------------------------- #
# Discovering runs
# --------------------------------------------------------------------------- #
def lang_of(path: Path) -> Optional[str]:
    parts = set(path.parts)
    for l in LANGS:
        if l in parts:
            return l
    return None


class Run:
    """One pipeline run = one leaf dir holding results_*.json. The _ours and
    _spider files share a byte-identical `results` array, so we read one."""

    def __init__(self, dirpath: Path, results: List[dict], lang: str, config: str):
        self.dirpath = dirpath
        self.results = results
        self.lang = lang
        self.config = config


def discover_runs(root: Path) -> List[Run]:
    runs: List[Run] = []
    for dirpath, _dirs, files in os.walk(root):
        jsons = [f for f in files if f.startswith("results_") and f.endswith(".json")]
        if not jsons:
            continue
        # prefer the _spider file (identical results either way)
        pick = next((f for f in jsons if f.endswith("_spider.json")),
                    next((f for f in jsons if f.endswith("_ours.json")), jsons[0]))
        d = Path(dirpath)
        try:
            data = json.loads((d / pick).read_text())
        except Exception as e:
            print(f"  WARN could not read {d/pick}: {e}", file=sys.stderr)
            continue
        lang = lang_of(d) or "?"
        runs.append(Run(d, data.get("results", []), lang, d.name))
    return runs


# --------------------------------------------------------------------------- #
# Metric + token computation
# --------------------------------------------------------------------------- #
def _mean(xs: List[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def compute_row(results: List[dict], view: Tuple, func_only_ids: Optional[set]) -> Optional[dict]:
    """Metrics AND token usage over ONE evaluated set: succeeded instances with a
    non-empty GT for `view` (∩ func_only when given). Tokens use the same set as
    the metrics, so a merged row has a single denominator `n` -- i.e. "tokens per
    evaluated instance". `out` = visible output = completion - reasoning; per-call
    columns = per-instance / calls_per_inst."""
    _label, gt_field, mp, bp = view
    ok = [r for r in results if not r.get("failed")]
    ev = [r for r in ok if r.get(gt_field)]
    if func_only_ids is not None:
        ev = [r for r in ev if r["instance_id"] in func_only_ids]
    n = len(ev)
    if n == 0:
        return None
    row = {
        "n": n,
        "recall":    _mean(r[f"{mp}recall"] for r in ev),
        "precision": _mean(r[f"{mp}precision"] for r in ev),
        "hit_rate":  _mean(r[f"{mp}hit_rate"] for r in ev),
        "accuracy":  _mean(r[f"{mp}accuracy"] for r in ev),
    }
    # dense-retrieval baseline over the same evaluated set (if present)
    if ev and f"{bp}recall" in ev[0]:
        row["bl_recall"]   = _mean(r[f"{bp}recall"] for r in ev)
        row["bl_hit_rate"] = _mean(r[f"{bp}hit_rate"] for r in ev)
        row["bl_accuracy"] = _mean(r[f"{bp}accuracy"] for r in ev)
        row["lift_recall"] = row["recall"] - row["bl_recall"]

    # token usage over the same evaluated set
    s_in = s_out_api = s_reason = s_calls = 0
    for r in ev:
        u = r.get("usage", {}) or {}
        s_in += u.get("prompt_tokens", 0)
        s_out_api += u.get("completion_tokens", 0)          # API output_tokens (incl reasoning)
        s_reason += u.get("reasoning_tokens", 0)            # 0 if never recorded
        s_calls += u.get("num_calls", 0)
    s_vis = s_out_api - s_reason                            # visible output only
    # max_ctx = peak single-call prompt tokens over the eval set (the context
    # window the model must support; per-call means hide this peak).
    max_ctx = 0
    for r in ev:
        for rd in r.get("rounds", []):
            # spider shape: one usage per center call in the round
            for c in rd.get("step_b_per_center", []):
                max_ctx = max(max_ctx, (c.get("usage", {}) or {}).get("prompt_tokens", 0))
            # agent shape: one usage per round (turn)
            ru = rd.get("usage")
            if ru:
                max_ctx = max(max_ctx, ru.get("prompt_tokens", 0))
    pi = lambda x: x / n
    pc = lambda x: (x / s_calls) if s_calls else 0.0
    row.update({
        "calls_per_inst": s_calls / n,
        "in_pi": pi(s_in), "out_pi": pi(s_vis), "reason_pi": pi(s_reason),
        "inout_pi": pi(s_in + s_vis), "total_pi": pi(s_in + s_vis + s_reason),
        "in_pc": pc(s_in), "out_pc": pc(s_vis), "reason_pc": pc(s_reason),
        "inout_pc": pc(s_in + s_vis), "total_pc": pc(s_in + s_vis + s_reason),
        "max_ctx": max_ctx,
    })
    return row


def macro(rows: List[dict], keys: List[str]) -> dict:
    """Unweighted mean of each key across rows (skips missing keys)."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = _mean(vals) if vals else None
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
# Combined table columns: metrics + a compact per-instance token block.
# in+out and in+out+reason are omitted (sums of in/out/reason); per-call = the
# per-instance value / calls_per_inst. Full 5x2 token detail lives in summary.csv.
M_COLS = [("recall", "recall", 3), ("precision", "prec", 3), ("hit_rate", "hit", 3),
          ("accuracy", "acc", 3), ("bl_recall", "dense_rec", 3), ("bl_hit_rate", "dense_hit", 3),
          ("bl_accuracy", "dense_acc", 3), ("lift_recall", "lift", 3),
          ("calls_per_inst", "calls/i", 2),
          ("in_pi", "in/inst", 1), ("out_pi", "out/inst", 1), ("reason_pi", "reason/inst", 1),
          ("in_pc", "in/call", 1), ("out_pc", "out/call", 1), ("max_ctx", "max_ctx", 0)]

# Full token quantities (per instance + per call) — CSV only.
CSV_TOK = ["calls_per_inst",
           "in_pi", "out_pi", "inout_pi", "reason_pi", "total_pi",
           "in_pc", "out_pc", "inout_pc", "reason_pc", "total_pc", "max_ctx"]


def short_config(config: str) -> str:
    """Compact the verbose run-dir name for display / narrow columns.
    'spider_edges-contains4+invokes2_nbrcap200' -> 'contains4+invokes2 nc200'."""
    return config.replace("spider_edges-", "").replace("_nbrcap", " nc")


def _fmt(v, w, prec=3):
    if v is None:
        return f"{'n/a':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.{prec}f}"
    return f"{v:>{w}}"


def _cfg_width(grouped: Dict[str, list]) -> int:
    return max([len("config")] + [len(short_config(c)) for c in grouped]) + 2


def _colw(label: str) -> int:
    return max(11, len(label) + 2)


def render_table(title: str, grouped: Dict[str, List[Tuple[str, dict]]]) -> str:
    """One combined table: metrics + per-instance tokens. grouped: config ->
    list of (lang, row). MACRO row per config when >1 lang."""
    cw = _cfg_width(grouped)
    lines = [title]
    header = f"  {'config':<{cw}}{'lang':<12}{'n':>5}" + "".join(f"{lab:>{_colw(lab)}}" for _, lab, _ in M_COLS)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for config in sorted(grouped):
        rows = grouped[config]
        lbl = short_config(config)
        for lang, r in sorted(rows, key=lambda x: x[0]):
            cells = "".join(_fmt(r.get(k), _colw(lab), p) for k, lab, p in M_COLS)
            lines.append(f"  {lbl:<{cw}}{lang:<12}{r['n']:>5}{cells}")
        if len(rows) > 1:
            mac = macro([r for _, r in rows], [k for k, _, _ in M_COLS])
            cells = "".join(_fmt(mac.get(k), _colw(lab), p) for k, lab, p in M_COLS)
            lines.append(f"  {lbl:<{cw}}{'MACRO-AVG':<12}{'':>5}{cells}")
    lines.append("")
    return "\n".join(lines)


def write_summary(d: Path, runs: List[Run], meta: Dict[str, Dict[str, dict]]) -> None:
    """Write SUMMARY.txt + summary.csv for directory d, covering `runs`."""
    n_langs = len({r.lang for r in runs})
    n_configs = len({r.config for r in runs})

    txt: List[str] = []
    txt.append("=" * 100)
    txt.append(f"SPIDER RESULTS SUMMARY   dir: {d}")
    txt.append(f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}   "
               f"covers {len(runs)} run(s) / {n_langs} lang(s) / {n_configs} config(s)")
    txt.append("=" * 100)
    txt.append("Metrics + tokens per GT view (spider = modified_nodes∩pool ; ours = LocAgent), "
               "per filter (all | func_only=no class-body edits).")
    txt.append("dense_* = no-LLM top-L retrieval baseline; lift = recall - dense_rec. "
               "Tokens are per EVALUATED instance (same set as metrics; n).")
    txt.append("out = visible output (completion - reasoning); reasoning=0 where not recorded. "
               "per-call = per-inst / calls/i. in+out, in+out+reasoning, per-call: see summary.csv.")
    txt.append("")

    csv_rows: List[dict] = []

    # ---- one combined table per (view, filter) ----
    for view in VIEWS:
        vlabel = view[0]
        for filt in FILTERS:
            grouped: Dict[str, List[Tuple[str, dict]]] = {}
            for run in runs:
                fids = None
                if filt == "func_only":
                    lang_meta = meta.get(run.lang, {})
                    fids = {i for i, m in lang_meta.items() if m.get("is_func_only")}
                row = compute_row(run.results, view, fids)
                if row is None:
                    continue
                grouped.setdefault(run.config, []).append((run.lang, row))
                csv_rows.append({
                    "gt_view": vlabel, "filter": filt,
                    "lang": run.lang, "config": run.config,
                    **{k: row.get(k) for k in
                       ["n", "recall", "precision", "hit_rate", "accuracy",
                        "bl_recall", "bl_hit_rate", "bl_accuracy", "lift_recall"]},
                    **{f"tok_{k}": row.get(k) for k in CSV_TOK},
                })
            if grouped:
                txt.append(render_table(
                    f"### GT view: {vlabel.upper()}   filter: {filt}", grouped))

    (d / "SUMMARY.txt").write_text("\n".join(txt))

    # ---- csv (full token detail: all 5 quantities x per-instance & per-call) ----
    fields = (["gt_view", "filter", "lang", "config", "n",
               "recall", "precision", "hit_rate", "accuracy",
               "bl_recall", "bl_hit_rate", "bl_accuracy", "lift_recall"]
              + [f"tok_{k}" for k in CSV_TOK])
    with open(d / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r.get(k) for k in fields})


# --------------------------------------------------------------------------- #
def main():
    global PICKLE_DIR, CACHE_FILE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=str, default=str(SCRIPT_DIR),
                    help="root dir to summarize (default: this script's dir)")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="rebuild the is_func_only metadata cache from the score pickles")
    ap.add_argument("--pickle-dir", type=str, default=str(DEFAULT_PICKLE_DIR),
                    help="dir holding test_<lang>_issue_node_scores_bfs4_filtered500.pkl, "
                         "the source of the is_func_only flag")
    ap.add_argument("--cache-file", type=str, default=str(CACHE_FILE),
                    help="where to store the is_func_only cache")
    args = ap.parse_args()

    PICKLE_DIR = Path(args.pickle_dir)
    CACHE_FILE = Path(args.cache_file)

    root = Path(args.root).resolve()
    meta = load_metadata_cache(rebuild=args.rebuild_cache)

    all_runs = discover_runs(root)
    if not all_runs:
        print(f"no result JSONs found under {root}", file=sys.stderr)
        return
    print(f"found {len(all_runs)} runs under {root}", file=sys.stderr)

    # Every directory at or below root that has >=1 run in its subtree gets a summary.
    dirs_with_runs = set()
    for run in all_runs:
        p = run.dirpath
        while True:
            dirs_with_runs.add(p)
            if p == root:
                break
            if root not in p.parents:
                break
            p = p.parent

    for d in sorted(dirs_with_runs):
        covered = [r for r in all_runs if d == r.dirpath or d in r.dirpath.parents]
        write_summary(d, covered, meta)
        print(f"  wrote {d/'SUMMARY.txt'}  ({len(covered)} runs)", file=sys.stderr)

    print(f"done: {len(dirs_with_runs)} directory summaries", file=sys.stderr)


if __name__ == "__main__":
    main()
