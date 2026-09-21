# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2: score every issue against its repository's graph nodes.

For each instance, computes a dense similarity and a BM25 score between the
issue's problem statement and every code-bearing node, then stores both rankings
plus the ground truth derived from the gold patch.

Two input formats are accepted:
  pkl (*.node_embeddings.pkl)  what data_prep.embed_graph_nodes writes. The
      issue embedding is computed here from the problem statement.
  npz (*.npz)                  arrays node_ids / code_embeddings / scores
      [/ query_embeddings]. With --use_precomputed_scores the stored `scores`
      array is reused verbatim and no embedding model is loaded -- this is the
      path the released numbers used, on the embeddings shipped with SpIDER-Bench.

Ground truth: `gt_files` / `gt_classes` / `gt_funcs` are the graph nodes whose
line span overlaps a line the gold patch edits (the "LocAgent GT" view). The
second GT view used in evaluation, SpIDER's `modified_nodes`, is not computed
here -- it is read straight out of the dataset's own metadata at scoring time.

Output: a single pickle, {"issues": {instance_id: {...}}, ...}. See the
"Score pickle schema" section of ../README.md.

Usage:
    # from *.node_embeddings.pkl produced by stage 1
    python -m data_prep.compute_issue_node_scores \
        --embeddings_dir data/node_embeddings/swepolybench/python \
        --graphs_dir     data/SpIDER-Bench/SWE-PolyBench/python \
        --dataset AmazonScience/SWE-PolyBench --split test \
        --output_path data/node_embeddings/swepolybench/test_python_issue_node_scores.pkl \
        --model_name Salesforce/SweRankEmbed-Small --backend hf \
        --device auto --max_length 2048 --trust_remote_code

    # from the *.npz embeddings shipped with SpIDER-Bench (no model loaded)
    python -m data_prep.compute_issue_node_scores \
        --embeddings_dir data/node_embeddings/swepolybench/python/<npz_dir> \
        --graphs_dir     data/SpIDER-Bench/SWE-PolyBench/python \
        --dataset AmazonScience/SWE-PolyBench --split test \
        --output_path data/node_embeddings/swepolybench/test_python_issue_node_scores.pkl \
        --embeddings_format npz --use_precomputed_scores
"""

import argparse
import json
import os.path as osp
import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import bm25s
import numpy as np
import Stemmer
from datasets import load_dataset
from tqdm.auto import tqdm
from unidiff.errors import UnidiffParseError

from .parse_patch import parse_patch


EMB_SUFFIX_PKL = ".node_embeddings.pkl"
EMB_SUFFIX_NPZ = ".npz"


def detect_embeddings_format(embeddings_dir: str) -> str:
    """Auto-detect whether embeddings_dir holds *.node_embeddings.pkl or *.npz files."""
    has_pkl = any(Path(embeddings_dir).glob(f"*{EMB_SUFFIX_PKL}"))
    has_npz = any(Path(embeddings_dir).glob(f"*{EMB_SUFFIX_NPZ}"))
    if has_pkl and has_npz:
        raise ValueError(
            f"{embeddings_dir} contains both *.node_embeddings.pkl and *.npz files; "
            "pass --embeddings_format explicitly."
        )
    if has_pkl:
        return "pkl"
    if has_npz:
        return "npz"
    raise FileNotFoundError(
        f"No *.node_embeddings.pkl or *.npz files found in {embeddings_dir}"
    )


def list_embedding_files(embeddings_dir: str, embeddings_format: str) -> List[str]:
    suffix = EMB_SUFFIX_NPZ if embeddings_format == "npz" else EMB_SUFFIX_PKL
    return sorted(str(p) for p in Path(embeddings_dir).glob(f"*{suffix}"))


def instance_id_from_path(path: str, embeddings_format: str) -> str:
    name = Path(path).name
    suffix = EMB_SUFFIX_NPZ if embeddings_format == "npz" else EMB_SUFFIX_PKL
    return name[: -len(suffix)] if name.endswith(suffix) else name


def load_embedding_payload(emb_file: str, embeddings_format: str,
                           allow_pickled_npz: bool = False):
    """Load an embeddings file into a normalized dict, or None if it is unreadable.

    Returned dict keys:
      node_ids           list[str]
      embeddings         np.ndarray (num_nodes, dim) float32 -- code-node embeddings
      precomputed_scores np.ndarray (num_nodes,) float32, or None
      query_embedding    np.ndarray (dim,) float32, or None
      graph_file         str, or None

    The two formats differ as follows:
      pkl (*.node_embeddings.pkl) -- dict with 'embeddings', 'node_ids', 'graph_file',
          'model_name'/'backend'; no query vector or scores (computed downstream).
      npz (*.npz)                 -- arrays 'code_embeddings', 'node_ids', 'scores'
          (precomputed dense cosine similarity) and usually 'query_embeddings'; no
          graph_file/model_name (graphs must be supplied via --graphs_dir).
    A corrupt/truncated npz returns None so the caller can skip it.

    npz files are read with allow_pickle=False: unpickling executes code embedded in
    the file, so a hostile .npz would be arbitrary code execution here. Every array
    read below is numeric or a string array, so a well-formed payload never needs it.
    A file storing *object* arrays cannot be read without unpickling -- that raises
    rather than being reported as corrupt, because skipping it would silently drop the
    instance from the output. `allow_pickled_npz` (--allow_pickled_npz) opts back in
    for files whose origin you trust.
    """
    if embeddings_format == "npz":
        # A real .npz is a zip of arrays. Checking that first keeps a truncated
        # download in the skip-and-warn path below: numpy would otherwise read a
        # non-zip as a bare .npy and report it as pickled data, which is
        # indistinguishable by message from a genuine object array.
        if not zipfile.is_zipfile(emb_file):
            print(f"  [warn] not a valid npz (truncated download?): {emb_file}")
            return None
        try:
            with np.load(emb_file, allow_pickle=allow_pickled_npz) as data:
                keys = set(data.files)
                node_ids = [str(x) for x in data["node_ids"]] if "node_ids" in keys else []
                embeddings = (
                    np.asarray(data["code_embeddings"], dtype=np.float32)
                    if "code_embeddings" in keys
                    else np.empty((0, 0), dtype=np.float32)
                )
                precomputed_scores = (
                    np.asarray(data["scores"], dtype=np.float32).reshape(-1)
                    if "scores" in keys
                    else None
                )
                query_embedding = (
                    np.asarray(data["query_embeddings"], dtype=np.float32).reshape(-1)
                    if "query_embeddings" in keys
                    else None
                )
            return {
                "node_ids": node_ids,
                "embeddings": embeddings,
                "precomputed_scores": precomputed_scores,
                "query_embedding": query_embedding,
                "graph_file": None,
            }
        except ValueError as exc:
            # Past the is_zipfile check, an allow_pickle complaint means a genuine
            # object array. That is a refusal, not a corrupt file: fail loudly rather
            # than silently skipping the instance, which would leave a near-empty
            # output pickle looking healthy.
            if not allow_pickled_npz and "allow_pickle" in str(exc):
                raise ValueError(
                    f"{emb_file} stores object arrays, which can only be read by "
                    f"unpickling -- i.e. by executing code supplied by whoever produced "
                    f"the file. Refusing by default. If you trust its origin, re-run with "
                    f"--allow_pickled_npz; otherwise re-save it with plain string/numeric "
                    f"dtypes (node_ids as '<U', embeddings/scores as float32)."
                ) from exc
            print(f"  [warn] failed to read embeddings file {emb_file}: {exc}")
            return None
        except Exception as exc:  # zip opens but a member is corrupt / unreadable
            print(f"  [warn] failed to read embeddings file {emb_file}: {exc}")
            return None

    with open(emb_file, "rb") as f:
        payload = pickle.load(f)
    return {
        "node_ids": list(payload.get("node_ids", [])),
        "embeddings": np.asarray(payload.get("embeddings", []), dtype=np.float32),
        "precomputed_scores": None,
        "query_embedding": None,
        "graph_file": payload.get("graph_file"),
    }


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def normalize_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_hf_query_embedder(model_name: str, device: str, max_length: int, trust_remote_code: bool):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Hugging Face query embedding requires transformers. Install with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        safe_serialization=True,  # repo ships .safetensors only; nomic_bert loader defaults to .bin
    )
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    import inspect
    try:
        forward_params = inspect.signature(model.forward).parameters
        supports_use_cache = "use_cache" in forward_params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()
        )
    except (TypeError, ValueError):
        supports_use_cache = False
    extra_forward_kwargs = {"use_cache": False} if supports_use_cache else {}

    def embed_query(text: str) -> np.ndarray:
        with torch.no_grad():
            encoded = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded, **extra_forward_kwargs)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vec = pooled.detach().cpu().numpy().astype(np.float32)[0]
            return vec

    return embed_query


def infer_model_info(embedding_file: str) -> Tuple[str, str]:
    with open(embedding_file, "rb") as f:
        payload = pickle.load(f)
    model_name = payload.get("model_name")
    backend = payload.get("backend", "hf")
    if not model_name:
        raise ValueError(f"Could not infer model_name from {embedding_file}")
    return model_name, backend


def collect_node_texts_from_graph(graph, node_ids: Sequence[str]) -> List[str]:
    texts = []
    for node_id in node_ids:
        node_data = graph.nodes.get(node_id, {})
        code = node_data.get("code")
        if not isinstance(code, str):
            code = ""
        texts.append(code)
    return texts


def extract_edited_lines_by_file(patch_text: str) -> Dict[str, set]:
    edited_lines_by_file: Dict[str, set] = {}
    if not patch_text:
        return edited_lines_by_file

    parsed = parse_patch(patch_text)
    for file_patch in parsed:
        file_path = file_patch.get("file")
        if not file_path:
            continue
        edited_lines_by_file.setdefault(file_path, set())
        for hunk in file_patch.get("hunks", []):
            changes = hunk.get("changes", {})
            for key in ("add", "delete"):
                for change in changes.get(key, []):
                    line_no = change.get("line")
                    if isinstance(line_no, int) and line_no > 0:
                        edited_lines_by_file[file_path].add(line_no)
    return edited_lines_by_file


def compute_gt_nodes_from_patch(graph, patch_text: str):
    try:
        edited_lines_by_file = extract_edited_lines_by_file(patch_text)
    except UnidiffParseError:
        return [], [], [], [], []

    if not edited_lines_by_file:
        return [], [], [], [], []

    gt_nodes = []
    gt_files = []
    gt_classes = []
    gt_funcs = []

    for file_path, edited_lines in edited_lines_by_file.items():
        if file_path in graph.nodes and graph.nodes[file_path].get("type") == "file":
            gt_files.append(file_path)
            gt_nodes.append(file_path)

        if not edited_lines:
            continue

        for nid, attrs in graph.nodes(data=True):
            ntype = attrs.get("type")
            if not str(nid).startswith(f"{file_path}:"):
                continue

            start_line = attrs.get("start_line")
            end_line = attrs.get("end_line")
            if not isinstance(start_line, int) or not isinstance(end_line, int):
                continue

            # If any edited line falls within the node span, consider it a GT node.
            hit = any(start_line <= ln <= end_line for ln in edited_lines)
            if not hit:
                continue

            gt_nodes.append(str(nid))

            if ntype == "class":
                gt_classes.append(str(nid))
            elif ntype == "function":
                gt_funcs.append(str(nid))

    gt_nodes = sorted(set(gt_nodes))
    gt_files = sorted(set(gt_files))
    gt_classes = sorted(set(gt_classes))
    gt_funcs = sorted(set(gt_funcs))
    gt_node_ids = sorted(set(gt_files) | set(gt_classes) | set(gt_funcs))
    return gt_nodes, gt_node_ids, gt_files, gt_classes, gt_funcs


def compute_bm25_scores(node_texts: Sequence[str], query: str, stemmer) -> np.ndarray:
    corpus_tokens = bm25s.tokenize(
        list(node_texts),
        lower=True,
        stopwords="english",
        stemmer=stemmer,
        return_ids=False,
        show_progress=False,
        leave=False,
    )
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens, show_progress=False)

    query_tokens = bm25s.tokenize(
        query,
        lower=True,
        stopwords="english",
        stemmer=stemmer,
        return_ids=False,
        show_progress=False,
        leave=False,
    )
    return bm25.get_scores(query_tokens[0]).astype(np.float32)


def build_ranked_dict(node_ids: Sequence[str], dense_scores: np.ndarray, bm25_scores: np.ndarray, order: np.ndarray) -> Dict[str, Dict[str, float]]:
    ranked = {}
    for idx in order:
        nid = str(node_ids[int(idx)])
        ranked[nid] = {
            "dense_similarity": float(dense_scores[int(idx)]),
            "bm25_score": float(bm25_scores[int(idx)]),
        }
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each SWE-bench issue problem_statement, compute dense similarity and BM25 scores "
            "against all code-bearing graph nodes, and store sorted score dictionaries."
        )
    )
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument(
        "--graphs_dir",
        type=str,
        default="",
        help="If set, resolve graph files as {graphs_dir}/{instance_id}.pkl instead of using the path embedded in *.node_embeddings.pkl. Required for --embeddings_format npz.",
    )
    parser.add_argument(
        "--embeddings_format",
        type=str,
        default="auto",
        choices=["auto", "pkl", "npz"],
        help=(
            "Embedding file format inside --embeddings_dir. 'pkl' = *.node_embeddings.pkl, "
            "'npz' = *.npz (code_embeddings/node_ids/scores[/query_embeddings]). "
            "'auto' detects which suffix is present."
        ),
    )
    parser.add_argument(
        "--use_precomputed_scores",
        action="store_true",
        help=(
            "Only valid with --embeddings_format npz. When set, reuse the dense-similarity "
            "'scores' array (and 'query_embeddings' issue vector) stored in each *.npz file "
            "and skip loading any embedding model. When NOT set, the issue embedding and "
            "dense scores are recomputed from the problem statement with --model_name."
        ),
    )
    parser.add_argument(
        "--allow_pickled_npz",
        action="store_true",
        help=(
            "Read *.npz embeddings with allow_pickle=True. Only needed for files storing "
            "object arrays, and it lets the file execute arbitrary code as it loads -- "
            "set it only for files whose origin you trust. Off by default."
        ),
    )
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--backend", type=str, default="", choices=["", "hf"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_issues", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=0, help="If >0, truncate ranked lists to top_k entries.")
    args = parser.parse_args()

    embeddings_format = args.embeddings_format
    if embeddings_format == "auto":
        embeddings_format = detect_embeddings_format(args.embeddings_dir)

    embedding_files = list_embedding_files(args.embeddings_dir, embeddings_format)
    if not embedding_files:
        suffix = EMB_SUFFIX_NPZ if embeddings_format == "npz" else EMB_SUFFIX_PKL
        raise FileNotFoundError(f"No *{suffix} found in {args.embeddings_dir}")

    use_precomputed_scores = args.use_precomputed_scores
    if use_precomputed_scores and embeddings_format != "npz":
        print("[warn] --use_precomputed_scores only applies to --embeddings_format npz; ignoring.")
        use_precomputed_scores = False

    if embeddings_format == "npz" and not args.graphs_dir:
        raise ValueError(
            "--graphs_dir is required with --embeddings_format npz "
            "(npz files do not store a graph_file path)."
        )

    # The query embedder (and the embedding model it loads) is only needed when we
    # have to compute the issue embedding / dense scores ourselves. With npz +
    # --use_precomputed_scores we reuse the stored 'scores' array and skip the model.
    need_query_embedder = not (embeddings_format == "npz" and use_precomputed_scores)

    if embeddings_format == "pkl":
        inferred_model_name, inferred_backend = infer_model_info(embedding_files[0])
    else:
        inferred_model_name, inferred_backend = "", "hf"
    model_name = args.model_name or inferred_model_name
    backend = args.backend or inferred_backend

    embed_query = None
    if need_query_embedder:
        if not model_name:
            raise ValueError(
                "Computing the issue embedding and dense scores requires an embedding "
                "model, but no --model_name was given and none could be inferred. Pass "
                "--model_name, or use --embeddings_format npz with --use_precomputed_scores."
            )
        if backend != "hf":
            raise ValueError(f"unknown backend {backend!r} (only 'hf' is supported)")
        embed_query = build_hf_query_embedder(
            model_name=model_name,
            device=args.device,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
        )

    ds = load_dataset(args.dataset, split=args.split)
    end_idx = len(ds) if args.end_idx < 0 else min(args.end_idx, len(ds))
    start_idx = max(0, args.start_idx)
    if end_idx < start_idx:
        raise ValueError(f"Invalid range: start_idx={args.start_idx}, end_idx={args.end_idx}")

    selected_indices = list(range(start_idx, end_idx))
    if args.max_issues > 0:
        selected_indices = selected_indices[: args.max_issues]
    selected = ds.select(selected_indices)

    emb_by_instance = {
        instance_id_from_path(p, embeddings_format): p for p in embedding_files
    }

    stemmer = Stemmer.Stemmer("english")
    results = {}
    missing_embeddings = []
    missing_graphs = []
    corrupt_embeddings = []

    iterator = tqdm(selected, desc="Issues", leave=True)
    for issue in iterator:
        issue_fields = {k: issue[k] for k in issue.keys()}
        instance_id = issue["instance_id"]
        problem_statement = issue.get("problem_statement", "") or ""
        emb_file = emb_by_instance.get(instance_id)
        if not emb_file:
            missing_embeddings.append(instance_id)
            continue

        payload = load_embedding_payload(emb_file, embeddings_format,
                                         allow_pickled_npz=args.allow_pickled_npz)
        if payload is None:
            corrupt_embeddings.append(instance_id)
            continue

        if args.graphs_dir:
            graph_file = osp.join(args.graphs_dir, f"{instance_id}.pkl")
        else:
            graph_file = payload.get("graph_file")
        if not graph_file or not osp.exists(graph_file):
            missing_graphs.append(instance_id)
            continue

        with open(graph_file, "rb") as f:
            graph = pickle.load(f)

        gt_nodes, gt_node_ids, gt_files, gt_classes, gt_funcs = compute_gt_nodes_from_patch(
            graph=graph,
            patch_text=issue.get("patch", "") or "",
        )

        node_ids = payload["node_ids"]
        node_embeddings = payload["embeddings"]

        # Resolve the issue (query) embedding. With npz + --use_precomputed_scores we
        # reuse the stored 'query_embeddings' vector (absent in some npz files, e.g.
        # most SWE-PolyBench typescript ones); otherwise we recompute it from the
        # problem statement with the embedding model.
        if use_precomputed_scores:
            raw_qvec = payload.get("query_embedding")
            if raw_qvec is not None:
                raw_qvec = np.asarray(raw_qvec, dtype=np.float32)
        else:
            raw_qvec = embed_query(problem_statement).astype(np.float32)

        if len(node_ids) == 0 or node_embeddings.size == 0:
            results[instance_id] = {
                "instance_id": instance_id,
                "problem_statement": problem_statement,
                "issue_fields": issue_fields,
                "issue_embedding": raw_qvec.tolist() if raw_qvec is not None else [],
                "embedding_file": emb_file,
                "graph_file": graph_file,
                "gt_nodes": gt_nodes,
                "gt_node_ids": gt_node_ids,
                "gt_files": gt_files,
                "gt_classes": gt_classes,
                "gt_funcs": gt_funcs,
                "num_nodes": 0,
                "dense_ranked": {},
                "bm25_ranked": {},
            }
            continue

        if node_embeddings.shape[0] != len(node_ids):
            raise ValueError(
                f"Embedding rows mismatch for {instance_id}: embeddings={node_embeddings.shape[0]}, node_ids={len(node_ids)}"
            )

        # Dense similarity scores. With npz + --use_precomputed_scores we reuse the
        # stored 'scores' array (cosine similarity of the issue vs. node embeddings);
        # otherwise we compute them from the L2-normalized query and node embeddings.
        if use_precomputed_scores:
            precomputed_scores = payload.get("precomputed_scores")
            if precomputed_scores is None:
                raise ValueError(
                    f"--use_precomputed_scores set but the npz file for {instance_id} has "
                    "no 'scores' array; rerun without --use_precomputed_scores to recompute."
                )
            dense_scores = np.asarray(precomputed_scores, dtype=np.float32)
            if dense_scores.shape[0] != len(node_ids):
                raise ValueError(
                    f"Precomputed score count mismatch for {instance_id}: "
                    f"scores={dense_scores.shape[0]}, node_ids={len(node_ids)}"
                )
        else:
            qvec = normalize_vec(raw_qvec)
            node_embeddings_norm = normalize_rows(node_embeddings)
            dense_scores = node_embeddings_norm @ qvec

        node_texts = collect_node_texts_from_graph(graph, node_ids)
        bm25_scores = compute_bm25_scores(node_texts=node_texts, query=problem_statement, stemmer=stemmer)

        dense_order = np.argsort(-dense_scores)
        bm25_order = np.argsort(-bm25_scores)

        if args.top_k > 0:
            dense_order = dense_order[: args.top_k]
            bm25_order = bm25_order[: args.top_k]

        dense_ranked = build_ranked_dict(node_ids, dense_scores, bm25_scores, dense_order)
        bm25_ranked = build_ranked_dict(node_ids, dense_scores, bm25_scores, bm25_order)

        results[instance_id] = {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "issue_fields": issue_fields,
            "issue_embedding": raw_qvec.tolist() if raw_qvec is not None else [],
            "embedding_file": emb_file,
            "graph_file": graph_file,
            "gt_nodes": gt_nodes,
            "gt_node_ids": gt_node_ids,
            "gt_files": gt_files,
            "gt_classes": gt_classes,
            "gt_funcs": gt_funcs,
            "num_nodes": len(node_ids),
            "dense_ranked": dense_ranked,
            "bm25_ranked": bm25_ranked,
        }

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": model_name,
        "backend": backend,
        "embeddings_format": embeddings_format,
        "use_precomputed_scores": use_precomputed_scores,
        "num_issues_scored": len(results),
        "num_missing_embeddings": len(missing_embeddings),
        "missing_embeddings": missing_embeddings,
        "num_corrupt_embeddings": len(corrupt_embeddings),
        "corrupt_embeddings": corrupt_embeddings,
        "num_missing_graphs": len(missing_graphs),
        "missing_graphs": missing_graphs,
        "issues": results,
    }

    out_path = args.output_path
    out_dir = osp.dirname(out_path)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if out_path.endswith(".json"):
        with open(out_path, "w") as f:
            json.dump(output, f)
    else:
        with open(out_path, "wb") as f:
            pickle.dump(output, f)

    print(
        f"Saved issue-node scores to {out_path}. "
        f"format={embeddings_format}, use_precomputed_scores={use_precomputed_scores}, "
        f"scored={len(results)}, missing_embeddings={len(missing_embeddings)}, "
        f"corrupt_embeddings={len(corrupt_embeddings)}, "
        f"missing_graphs={len(missing_graphs)}"
    )


if __name__ == "__main__":
    main()
