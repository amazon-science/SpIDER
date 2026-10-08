# SpIDER — Spatially Informed Dense Embedding Retrieval

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026%20Main-1f6feb.svg)](https://2026.emnlp.org)
[![Paper](https://img.shields.io/badge/arXiv-2512.16956-b31b1b.svg)](https://arxiv.org/abs/2512.16956)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SpIDER--Bench-ffd21e.svg)](https://huggingface.co/datasets/AmazonScience/SpIDER-Bench)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**[SpIDER: Spatially Informed Dense Embedding Retrieval for Software Issue Localization](https://arxiv.org/abs/2512.16956)**<br>
Shravan Chaudhari, Rahul Thomas Jacob, Jiajun Cao, Shihab Rashid, Mononito Goswami, Christian Bock<br>
*EMNLP 2026 (Main Conference)*

> 🎉 SpIDER has been accepted to the **EMNLP 2026 Main Conference**. The
> [SpIDER-Bench](https://huggingface.co/datasets/AmazonScience/SpIDER-Bench) graphs are
> available on the Hugging Face Hub.

Reference implementation of the SpIDER issue-localization loop from our EMNLP 2026
paper ([arXiv:2512.16956](https://arxiv.org/abs/2512.16956)), plus the data-preparation
chain and the summarizer that produce the reported tables.

SpIDER starts from a dense retrieval ranking over a repository's code graph and
uses an LLM for exactly one kind of judgment: *does this graph neighbor itself
need to change?* Everything else — which nodes become exploration centers, how
the ranked window is rewritten — is deterministic.

Per round, for a window of the top-`K` ranked functions:

1. **Centers** = the top-`C` of the window. No LLM call.
2. **Neighbor filtering** — for each center, BFS its neighborhood in the code
   graph (per edge type, at that edge type's own depth), keep the neighbors that
   fall inside the top-`N` candidate pool, and ask the LLM to restrictively
   select the ones that must be modified. `C` calls.
3. **Window rewrite** — each selected neighbor is inserted directly below its
   center; an equal number of bottom-ranked nodes drop out to keep the window at
   size `K`.

The final prediction is the top-`L` of the accumulated window. With the released
defaults (`K=20 L=20 N=500 C=5 max_steps=1`) that is 5 LLM calls per instance.

---

## Contents

```
spider/                 the pipeline
  pipeline.py           the SpIDER loop (steps 1-3 above)
  data_loader.py        rankings, graphs, BFS, both ground-truth views
  prompt_format.py      prompt rendering + response parsing
  llm_client.py         OpenAI / Azure / Azure-Responses / vLLM transports
  result_builder.py     per-instance result dict, incl. the no-LLM baseline
  summary.py            aggregation + result serialization
  run_api.py            runner: hosted LLM endpoint, thread pool
  run_vllm.py           runner: local models, one vLLM server per GPU
  multigpu_utils.py     vLLM server launch/teardown
data_prep/              graphs -> embeddings -> scores -> candidate pool
scripts/
  download_bench.py     fetch SpIDER-Bench from the Hub and rebuild the graphs
  materialize_graphs.py parquet -> {instance_id}.pkl (the preprocessing step)
  spider_bench_schema.py  parquet schema + the graph<->tables conversion
  build_hf_dataset.py   producer side: pickle tree -> parquet (run once, by us)
  audit_graph_schema.py derives the parquet schema from every graph in the tree
  prepare_data.sh       run the three data-prep stages for one dataset item
  run_spider_api.sh     launcher for the hosted-API runner
  run_spider.sh         launcher for the local-vLLM runner
  reproduce_paper_runs.sh   every configuration behind the reported tables
summarize.py            result JSONs -> SUMMARY.txt / summary.csv
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. The core runtime is small (`openai`, `httpx`, `networkx`, `numpy`);
`requirements.txt` marks which extras belong to data preparation (`datasets`,
`bm25s`, `PyStemmer`, `unidiff`), to node embedding (`torch`, `transformers`),
and to the local-model runner (`vllm`).

---

## Data

SpIDER needs two things per benchmark instance:

| what | where it comes from |
|---|---|
| **repository graph** — `{instance_id}.pkl`, an `nx.MultiDiGraph` whose nodes carry `type` / `code` / `start_line` / `end_line` and whose edges carry `type` ∈ {`contains`, `imports`, `inherits`, `invokes`} | [SpIDER-Bench](https://huggingface.co/datasets/AmazonScience/SpIDER-Bench) on the Hub, rebuilt from parquet (below) |
| **score pickle** — the dense ranking over those nodes, plus ground truth | you build it (below), or reuse ours |

Both are Python pickles, which run code when loaded — see [Security](#security) before
loading either from a source you don't control.

### 1. Graphs

[SpIDER-Bench](https://huggingface.co/datasets/AmazonScience/SpIDER-Bench) is published as
**parquet**, so it loads with `datasets` and renders in the Hub dataset viewer. The pipeline wants `nx.MultiDiGraph` pickles, so downloading it is two
steps — download, then rebuild. `download_bench.py` does both:

```bash
python scripts/download_bench.py --out data/SpIDER-Bench
# or just what you need:
python scripts/download_bench.py --out data/SpIDER-Bench --subset SWE-PolyBench/python
```

**The rebuild is required.** Nothing downstream reads parquet; every stage resolves
`{graphs_dir}/{instance_id}.pkl`. If you fetch the parquet some other way, run the
preprocessing step yourself:

```bash
python scripts/materialize_graphs.py --out data/SpIDER-Bench                       # from the Hub
python scripts/materialize_graphs.py --out data/SpIDER-Bench --parquet-dir <dir>   # from a local copy
python scripts/materialize_graphs.py --out data/SpIDER-Bench \
    --instance-ids astropy__astropy-12907 django__django-11099                     # a few instances
```

It downloads only the shards holding what you asked for, skips instances whose pickle
already exists, and `--verify-against <original tree>` re-checks every rebuilt graph with
`networkx.utils.graphs_equal`.

Layout after the rebuild — what `--graphs_dir` expects:

```
data/SpIDER-Bench/SWE-PolyBench/{python,java,javascript,typescript}/{instance_id}.pkl
data/SpIDER-Bench/SWE-bench_Verified/{instance_id}.pkl
data/SpIDER-Bench/Multi-SWE-bench/{java,javascript,typescript}/{instance_id}.pkl
```

Budget ~7 GB for the parquet and ~73 GB for the rebuilt pickles.
`download_bench.py --no-keep-parquet` drops the former once the latter exist.

<details>
<summary>Parquet layout, if you want to read the graphs directly</summary>

Each graph is stored relationally — one row per node and one row per edge — because a
single instance reaches hundreds of MB, too large for a parquet row or the viewer. Three
configs: `instances` (a one-row-per-graph summary), `nodes` and `edges`; the latter two
carry one split per benchmark+language:

```python
from datasets import load_dataset
nodes = load_dataset("AmazonScience/SpIDER-Bench", "nodes", split="swe_polybench_python")
edges = load_dataset("AmazonScience/SpIDER-Bench", "edges", split="swe_polybench_python")
```

Rebuilding a graph faithfully means iterating `node_ord` / `edge_ord` (networkx insertion
order, which the BFS tie-breaks depend on) and reading attributes in `attr_order` order —
parquet null cannot distinguish *attribute absent* from *attribute present with value
`None`*, and both occur in this data. `scripts/spider_bench_schema.py` holds the
conversion; `tables_to_graph()` is the function to call.

</details>

### 2. Score pickles

Three stages, wrapped by `scripts/prepare_data.sh`:

```
stage 1  data_prep.embed_graph_nodes          graphs     -> {instance_id}.node_embeddings.pkl
stage 2  data_prep.compute_issue_node_scores  embeddings -> test_<lang>_issue_node_scores.pkl
stage 3  data_prep.add_filtered_bfs_rankings  scores     -> test_<lang>_issue_node_scores_bfs4_filtered500.pkl
```

**Bring your own embeddings.** Stage 1 embeds every code-bearing graph node with
any Hugging Face encoder (mean pooling, optional L2 normalization) — the reported
numbers use `Salesforce/SweRankEmbed-Small`. Point `--model_name` at whatever
encoder you want to evaluate and `--output_dir` at wherever you want the vectors
kept; nothing downstream cares about the location, only that stage 2 is pointed
at the same directory.

```bash
LANG=python MODEL_NAME=Salesforce/SweRankEmbed-Small NUM_GPUS=4 \
    ./scripts/prepare_data.sh
```

**Or bring your own scores and skip stage 1.** Stage 2 also accepts `*.npz`
embedding files holding `node_ids`, `code_embeddings`, `scores` (issue-vs-node
cosine similarity) and optionally `query_embeddings`. With
`--use_precomputed_scores` it reuses the stored `scores` array verbatim and loads
no model at all — no GPU needed. This is the path the released numbers were
produced on, from SweRankEmbed-Small embeddings computed once and cached as npz,
and it is also the cheapest way to plug in a retriever that already emits
issue-vs-node scores:

```bash
LANG=python STAGES="2 3" EMB_FORMAT=npz USE_PRECOMPUTED_SCORES=1 \
    EMBEDDINGS_DIR=/path/to/your/npz_dir \
    ./scripts/prepare_data.sh
```

The npz files are named `{instance_id}.npz`; stage 2 matches them to dataset rows
by that stem.

Stage 3 is **optional for running SpIDER**: the pipeline re-runs BFS live against
the graph at its own configured depth, so it reads `dense_ranked` but not the
`bfs_*` fields stage 3 writes. It is included because the released artifacts were
produced from its output, and because the `bfs_*` fields are what the
retrieval-ceiling analyses consume. If you skip it, point `SCORES_PATH_OVERRIDE`
at the stage-2 pickle.

> `data_prep.compute_issue_node_scores` pulls problem statements and gold patches
> from the Hub. On a machine without outbound network, pre-populate the dataset
> cache and export `HF_HUB_OFFLINE=1`, or the import hangs on a Hub round-trip.

### Score pickle schema

If you want to plug in your own retriever, produce this and skip `data_prep/`
entirely. Only the starred fields are read by the pipeline.

```python
{
  "issues": {
    "<instance_id>": {
      "problem_statement": str,           # * the issue text
      "dense_ranked": {                   # * ranked BEST FIRST; dict order IS the ranking
          "<node_id>": {"dense_similarity": float, "bm25_score": float}, ...
      },
      "bm25_ranked": {...},               #   same shape; used with --ranking_type bm25
      "graph_file": str,                  # * absolute path; falls back to
                                          #   {--graphs_dir}/{instance_id}.pkl if missing
      "gt_files":   [str], "gt_classes": [str], "gt_funcs": [str],
                                          # * LocAgent GT: nodes whose line span
                                          #   overlaps a line the gold patch edits
      "issue_fields": {                   #   raw dataset row
          "modified_nodes": str, ...      # * SpIDER GT source (SWE-PolyBench only)
      },
      # ... plus the bfs_* fields stage 3 adds
    }, ...
  },
}
```

`<node_id>` is `path/to/file.py` for a file, `path/to/file.py:Class` for a class,
and `path/to/file.py:Class.method` (or `path/to/file.py:function`) for a
function — the same IDs used as graph node keys.

---

## Running

### Hosted LLM endpoint (no GPUs)

Credentials come from the environment; nothing is hardcoded.

```bash
export LLM_API_KEY=...
export LLM_MODEL=<deployment-or-model-name>
export LLM_PROVIDER=azure_responses          # azure_responses | azure | openai | vllm
export LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1

LANGS=python LIMIT=5 ./scripts/run_spider_api.sh      # smoke test first
./scripts/run_spider_api.sh                          # all four languages
```

Useful knobs (all environment variables):

| var | default | meaning |
|---|---|---|
| `DATASET` | `swepolybench` | `swepolybench` or `swebench_verified` |
| `LANGS` | all four | which SWE-PolyBench languages to run |
| `EDGES` | `contains` | comma-separated: `contains,imports,inherits,invokes` |
| `BFS_DEPTH` | `4` | uniform depth for every edge type above |
| `BFS_DEPTH_PER_EDGE` | — | per-edge overrides, e.g. `contains:4,invokes:2` |
| `MAX_NEIGHBORS_PER_CENTER` | `200` | cap on neighbors shown per center; `0` = unlimited |
| `K` `L` `N` `C` | `20 20 500 5` | window / budget / candidate pool / centers |
| `MAX_STEPS` | `1` | rounds of the loop |
| `MAX_TOKENS` | `8192` | max output tokens per call |
| `NUM_WORKERS` | `8` | concurrent API calls; lower it if you hit 429s |
| `LIMIT` / `INSTANCE_IDS` | — | scope a trial run |
| `DATA_ROOT` `GRAPHS_ROOT` `SCORES_ROOT` `RESULTS_ROOT` | under the repo | data and output locations |

### Local models (one vLLM server per GPU)

```bash
GPUS="0 1 2 3" MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct ./scripts/run_spider.sh
./scripts/run_spider.sh --skip_server_launch     # servers already listening
```

Same pipeline, same output schema — only the LLM transport differs.

### Reproducing the reported configurations

```bash
./scripts/reproduce_paper_runs.sh     # ~33 runs; read the cost note at the top first
python summarize.py --root results/<model> \
    --pickle-dir data/node_embeddings/swepolybench --rebuild-cache
```

Each configuration writes to its own directory, so lines can be run, re-run, or
commented out independently.

---

## Output

```
results/<model>/<dataset>/<item>/spider_edges-<slug>[_nbrcap<n>]/
    results_<tag>_ours.json      LocAgent GT view
    results_<tag>_spider.json    SpIDER GT view (modified_nodes ∩ candidate pool)
```

Both files carry a byte-identical `results` array and differ only in the
`aggregate` block. `_spider.json` is written only when the dataset supplies
`modified_nodes` — SWE-PolyBench does, SWE-bench Verified does not, so Verified
runs produce the `ours` file alone.

Per instance the result records the final `selected` list, both GT views and
their recall / precision / hit-rate / accuracy, the **no-LLM dense-retrieval
baseline** at the same top-`L` budget (`baseline_*` / `spider_baseline_*` — the
control every reported lift is measured against), token usage, and the full
per-round trajectory: centers, neighbors shown, raw model response, and the
resolved picks per center.

### Reproducibility

Everything upstream of the LLM is deterministic: given the same score pickle and
graphs, the centers, the BFS neighborhoods, the prompts and the window rewrite
are reproduced exactly. What varies is the endpoint. Hosted reasoning models do
not honor `temperature=0` as a determinism guarantee — the same prompt can come
back with a different selection.

In a spot check of 9 instance-runs against the released artifacts (SWE-PolyBench
python and java, `contains@4` and `contains@4+invokes@2`), 8 reproduced the
published output byte for byte: identical centers, identical per-center prompt
token counts, identical model picks, identical final selections. The ninth had
identical prompts and ground truth but one of its five Step-B calls returned an
empty selection where the original had picked one neighbor — metrics for that
instance were unchanged. Expect aggregate numbers to land within sampling noise
rather than to match to the last decimal.

### Two things to check before trusting a run

`num_call_errors` and `num_empty_responses`, both aggregated in the run log. A
Step-B call that errors (usually context overflow) or returns no text at all
parses as an empty selection — indistinguishable from a confident "nothing here
needs changing". Either way, those centers contributed nothing and the run
partially fell back to the plain ranking baseline. Non-zero means raise
`MAX_TOKENS`, lower `MAX_NEIGHBORS_PER_CENTER`, or both.

Reasoning models make the empty case easy to hit: they spend `max_output_tokens`
on reasoning first, and a starved call returns reasoning and no answer.

### Summaries

```bash
python summarize.py --root results/<model> --pickle-dir data/node_embeddings/swepolybench
```

Writes `SUMMARY.txt` (fixed-width) and `summary.csv` (long-form) into every
directory at or below `--root` that has runs under it. Each is reported for both
GT views and under two instance filters: `all`, and `func_only` (patches that
touch only functions, no class-body edits). The `func_only` filter reads
`is_func_only` from the score pickles via a small cache; without the pickles
those tables are simply omitted and the `all` tables are unaffected.

---

## Choosing edge types

Each edge type expands independently from the seed at its own depth, and the
results are unioned — mixed-type paths (2 `contains` hops then 2 `invokes` hops)
are **not** traversed. That is what lets `invokes` stay capped at depth 2 while
`contains` runs at depth 4 without one explosion swamping the other.

Mean function-neighbors found per center, measured on SWE-PolyBench/python:

| edges | neighbors/center | note |
|---|---|---|
| `contains@4` | 11 | the default |
| `imports@4` | 5 | |
| `inherits@4` | 0 | finds **no** function-type neighbors — alone it degrades the run to the plain-ranking baseline. Only useful inside a union. |
| `invokes@2` | 34 | |
| `invokes@4` | 181 | ~70k tokens of code at `MAX_CODE_CHARS=1500`. Always pair depth > 2 `invokes` with `MAX_NEIGHBORS_PER_CENTER`. |

The `bfs4` in a score-pickle filename refers to how its 500-node candidate pool
was precomputed. It does **not** constrain `BFS_DEPTH` at run time, which is
re-run live against the graph.

---

## Security

**Pickled artifacts execute code when they are loaded.** `pickle.load` and
`numpy.load(..., allow_pickle=True)` rebuild objects by running instructions embedded
in the file, so opening a malicious `.pkl` is arbitrary code execution in the loading
process — not a parsing error you can catch. This is a property of the format, not of
this code.

Only load artifacts whose origin you trust. Specifically:

| input | loaded by |
|---|---|
| graph pickles `{instance_id}.pkl` | `spider/data_loader.py`, `data_prep/*`, `scripts/build_hf_dataset.py`, `scripts/audit_graph_schema.py`, and `scripts/materialize_graphs.py --verify-against` |
| score pickles | `spider/data_loader.py`, `data_prep.add_filtered_bfs_rankings`, `summarize.py` |
| `*.node_embeddings.pkl` | `data_prep.compute_issue_node_scores` |

Prefer **rebuilding the graphs locally from the published parquet** with
`scripts/download_bench.py` / `scripts/materialize_graphs.py` over accepting graph
pickles from anyone else. Parquet is pure data and cannot execute anything; the
pickles are then produced on your own machine.

`*.npz` embeddings are read with `allow_pickle=False`, so an untrusted npz cannot
execute code. A file that stores *object* arrays cannot be read that way and is
refused with an explanatory error rather than skipped; `--allow_pickled_npz` opts
back in for files you trust.

`--trust_remote_code` (in `data_prep.embed_graph_nodes` and
`data_prep.compute_issue_node_scores`) makes `transformers` execute Python shipped by
the Hugging Face repo named in `--model_name`. It is off by default on both modules and
should stay off unless you trust that repo. Note that `scripts/prepare_data.sh` does
pass it, for the default encoder. `--dataset` likewise pulls and runs the loading
script of a Hub dataset.

None of this is an argument for reproducing the paper differently — the released
artifacts are ours. It matters when someone hands you a `.pkl` and asks you to score it.

---

## Citation

If you use SpIDER or SpIDER-Bench, please cite our EMNLP 2026 paper:

```bibtex
@inproceedings{chaudhari2026spiderspatiallyinformeddense,
      title={SpIDER: Spatially Informed Dense Embedding Retrieval for Software Issue Localization},
      author={Shravan Chaudhari and Rahul Thomas Jacob and Jiajun Cao and Shihab Rashid and Mononito Goswami and Christian Bock},
      booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2026},
      url={https://arxiv.org/abs/2512.16956},
}
```

---

## Reporting security issues

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE](LICENSE) and [NOTICE](NOTICE).

`data_prep/parse_patch.py` and the ground-truth construction in
`data_prep/compute_issue_node_scores.py` are adapted from
[LocAgent](https://github.com/gersteinlab/LocAgent), which is also
Apache-2.0 licensed.
