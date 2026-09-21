# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research code for the SpIDER issue-localization paper (arXiv:2512.16956). `README.md` is the
user-facing manual and is detailed — read it for hyperparameters, env vars, and data layout.
This file covers what the README doesn't: cross-file invariants and where things break.

## Commands

```bash
pip install -r requirements.txt                    # Python 3.10+

# Data (see README "Data"; ~7 GB parquet, ~73 GB rebuilt pickles)
python scripts/download_bench.py --out data/SpIDER-Bench --subset SWE-PolyBench/python
LANG=python STAGES="2 3" EMB_FORMAT=npz USE_PRECOMPUTED_SCORES=1 \
    EMBEDDINGS_DIR=<npz_dir> ./scripts/prepare_data.sh

# Run (launchers are env-var driven; they wrap `python -m spider.run_api|run_vllm`)
LANGS=python LIMIT=5 ./scripts/run_spider_api.sh   # smoke test — always do this first
GPUS="0 1 2 3" MODEL=<hf-model> ./scripts/run_spider.sh
python summarize.py --root results/<model> --pickle-dir data/node_embeddings/swepolybench
```

**There is no test suite, linter, or build step** — no `pyproject.toml`, no `pytest`, no CI. Do not
invent one or claim tests pass. Verification is:

```bash
python -m compileall -q spider data_prep scripts summarize.py   # syntax
bash -n scripts/*.sh                                           # shell syntax
INSTANCE_IDS="<id from the score pickle>" LANGS=python ./scripts/run_spider_api.sh   # one instance
```

The closest thing to a unit test is a one-instance run: `--instance_ids` / `--limit` on the runners,
`INSTANCE_IDS` / `LIMIT` on the launchers. Any LLM call costs money, so scope with `LIMIT` first.

## Architecture

### The data contract is two artifacts joined by `instance_id`

Nothing reads parquet at run time. Every stage resolves `{graphs_dir}/{instance_id}.pkl`.

- **graph pickle** — one `nx.MultiDiGraph` per instance; nodes carry `type`/`code`/`start_line`/
  `end_line`, edges carry `type` ∈ {`contains`,`imports`,`inherits`,`invokes`}.
- **score pickle** — `{"issues": {instance_id: {...}}}`; schema in README. Only a handful of fields
  are actually read (`problem_statement`, `dense_ranked`, `graph_file`, `gt_*`,
  `issue_fields.modified_nodes`). To plug in a different retriever, emit this and skip `data_prep/`.

Two invariants that are easy to violate:

1. **`dense_ranked` dict insertion order *is* the ranking** (best first). Anything that round-trips
   it through a structure that doesn't preserve order silently destroys the ranking. Same reason
   `spider_bench_schema.py` stores `node_ord`/`edge_ord`/`attr_order` — networkx insertion order
   affects BFS tie-breaks, and parquet null can't distinguish *attribute absent* from *attribute
   present but `None`*, both of which occur in this data.
2. **`graph_file` in the pickle is the absolute path of the machine that built it** and normally
   doesn't exist. `data_loader._resolve_graph_file` falls back to `{graphs_dir}/{instance_id}.pkl`;
   that fallback is what makes the artifacts portable.

Node IDs: `path/file.py` (file), `path/file.py:Class` (class), `path/file.py:Class.method` or
`path/file.py:function` (function). Same strings as graph node keys.

### One LLM decision, everything else deterministic

`spider/pipeline.py` is the whole loop. The only LLM call site is `run_step_b_restrictive`, asking
per center: *does this graph neighbor itself need to change?* Centers (`window[:C]`), BFS
neighborhoods, prompts, and the window rewrite (`build_modified_window`: each selected neighbor
inserted directly below its center, bottom-ranked nodes dropped to hold `K`) are all rule-based.
Exactly `C` calls per round. When touching the pipeline, preserve this — it's the paper's claim.

`InstanceData` (`data_loader.py`) type-filters the full ranking down to `cfg.node_type`, and window
(`get_window`), neighbor pool (`get_neighbor_candidates`, ranks K+1..K+N), and BFS results are all
rank-ordered slices of that one list, threaded through a shared `seen` set so nothing is shown twice.

**BFS unions per edge type independently from the seed** — each edge expands at its own depth
(`cfg.edge_depth_map()`) and results are unioned; mixed-type paths are *never* traversed. That is
what lets `invokes` stay at depth 2 while `contains` runs at depth 4 without one swamping the other.
`adj_for_edge` caches undirected adjacency per (instance, edge type); it exists because rebuilding
it per frontier node dominated runtime on dense edges.

### Two ground-truth views, always computed side by side

- **LocAgent GT** (`gt_files`/`gt_classes`/`gt_funcs`) — read from the score pickle; graph nodes
  whose line span overlaps a line the gold patch edits.
- **SpIDER GT** (`spider_gt_*`) — derived in `_build_spider_gt` from SWE-PolyBench `modified_nodes`,
  intersected with the candidate pool. SWE-bench Verified has no `modified_nodes`, so this is empty
  and only the `_ours.json` file is written.

Every metric therefore exists twice (`recall` / `spider_recall`, …) through `result_builder.py` →
`summary.build_views` → two JSON files with a **byte-identical `results` array differing only in
`aggregate`**. Add a metric to one view and you must add it to both, plus its `baseline_` twin.

`result_builder` also computes the **no-LLM control**: `ranked_nodes[:L]`, the plain-retrieval
baseline at the same budget. Every reported lift is measured against it, so it is not optional
bookkeeping. `summary.build_agg` omits the baseline block entirely when absent rather than reporting
zeros — a 0.0 would read as a real measurement.

Aggregation divides eval metrics by *evaluated* instances (succeeded AND non-empty GT for the
granularity), but process stats (tokens, rounds, time) by all succeeded. Don't unify these.

### Empty selections are ambiguous — that's why the flags exist

A Step-B call that errors (usually context overflow) or returns no text parses as "no neighbors need
changing", indistinguishable from a confident empty pick. Either way those centers contribute
nothing and the run silently degrades toward the plain-ranking baseline. Hence `call_error` /
`empty_response` per center, `num_call_errors` / `num_empty_responses` aggregated, and a loud warning
in both runners. **Check these before trusting any run.** Reasoning models hit this easily: they
spend `max_tokens` on reasoning and return no answer. Never "simplify" these flags away.

### Output paths and secrets

`cfg.run_slug()` (`config.py`) is the single source of truth for both the results filename tag and
the output subdirectory, so two edge configs can't collide. `summary.resolve_out_dir` appends it to
the leaf dir and **mutates `cfg.output_dir`** so the serialized config names the directory actually
written.

`vars(cfg)` goes into every results JSON, so `summary._SECRET_FIELDS` masks the API-key fields. Any
new credential field must be added there. Keys come from the environment (`LLM_API_KEY`,
`AZURE_API_KEY`) — never hardcoded, never committed.

### Provider abstraction and the two runners

`llm_client.py` exposes one `chat(messages) -> LLMResponse` across OpenAI / Azure native / Azure
Responses / vLLM, so pipeline code never branches on provider. `completion_tokens` **includes**
reasoning tokens; `reasoning_tokens` is a subset (0 for non-reasoning models and vLLM). Transient
failures (429/timeout/5xx) retry with backoff inside the client; anything else propagates and is
recorded as a Step-B `call_error`.

`run_api.py` (thread pool, shared client) and `run_vllm.py` (one vLLM server per GPU, round-robin
shards) share `pipeline`, `summary`, and `run_api._failed_result` — only transport differs, and
output schemas must stay identical. `_failed_result` deliberately mirrors the success schema field
for field so failures aggregate as ordinary rows; extend both together. Both stream a
`.partial.jsonl` during the run and delete it on success.

## Conventions

- Adding a hyperparameter touches four places: `SelectorConfig`, both runners' `argparse`, and the
  launcher shell script. Missing the launcher makes it unreachable in practice.
- Launchers are pure env-var interfaces with `${VAR:-default}` and skip-with-warning when a
  `scores_path`/`graphs_dir` is missing — a missing language shouldn't abort a four-language sweep.
- Malformed `--bfs_depth_per_edge` raises rather than silently dropping an override
  (`parse_edge_depths`); keep typos loud.
- Comments here explain *why* (empirical costs, failure modes, reproducibility constraints), often
  with measured numbers. Match that when editing; don't strip them as noise.
- Apache-2.0: every `.py`/`.sh` starts with the two-line copyright + `SPDX-License-Identifier`
  header (after the shebang). New source files need it. No author lines.

## Gotchas

- `data_prep.compute_issue_node_scores` hits the HF Hub for problem statements and gold patches. On
  an offline machine, pre-populate the cache and export `HF_HUB_OFFLINE=1`, or the import hangs.
- Every `.pkl` here executes code on load (see README "Security"). `.npz` is deliberately read with
  `allow_pickle=False`; don't "fix" that back. Object-array npz files raise a pointed error instead
  of returning `None`, because `None` means *skip this instance* and would leave a near-empty score
  pickle looking healthy — `--allow_pickled_npz` is the explicit opt-in.
- Stage 3 (`add_filtered_bfs_rankings`) is **optional** for running SpIDER — the pipeline re-runs BFS
  live and reads only `dense_ranked`. Point `SCORES_PATH_OVERRIDE` at the stage-2 pickle to skip it.
- `bfs4` in a score-pickle filename describes how its 500-node pool was precomputed. It does **not**
  constrain `BFS_DEPTH` at run time.
- `invokes` at depth > 2 must be paired with `MAX_NEIGHBORS_PER_CENTER` (~181 neighbors/center,
  ~70k tokens). `inherits` alone finds no function-type neighbors and degrades to the baseline.
- Everything upstream of the LLM reproduces exactly; hosted reasoning models do not honor
  `temperature=0` as a determinism guarantee. Expect aggregates within sampling noise, not
  bit-identical numbers.
