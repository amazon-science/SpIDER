# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1: embed every code-bearing node of every repository graph.

Reads the per-instance graph pickles, embeds each node's `code` attribute with a
Hugging Face encoder, and writes one `{instance_id}.node_embeddings.pkl` per
graph. Any HF model with a standard encoder head works -- the released numbers
use Salesforce/SweRankEmbed-Small with mean pooling and L2 normalization.

Instances of the same repository are processed consecutively and share a cache
keyed by node-code hash, so unchanged nodes are embedded once per repo rather
than once per instance. That is the difference between hours and days on
SWE-PolyBench, where a repo can appear in hundreds of instances.

Output schema (per file):
    graph_file   str          path of the source graph pickle
    model_name   str
    backend      str
    node_ids     list[str]    node IDs, aligned with `embeddings` rows
    node_types   list[str]    "file" | "class" | "function"
    start_lines  list[int]
    end_lines    list[int]
    embeddings   np.ndarray   (num_nodes, dim) float32

Usage:
    python -m data_prep.embed_graph_nodes \
        --graph_dir data/SpIDER-Bench/SWE-PolyBench/python \
        --model_name Salesforce/SweRankEmbed-Small \
        --output_dir data/node_embeddings/swepolybench/python \
        --batch_size 128 --num_gpus 4 --normalize --trust_remote_code
"""

import argparse
import hashlib
import os
import os.path as osp
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch.multiprocessing as mp
from tqdm.auto import tqdm


def list_graph_files(graph_dir: str) -> List[str]:
    return sorted(str(p) for p in Path(graph_dir).glob("*.pkl"))


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_hf_embedder(
    model_name: str,
    device: str,
    max_length: int,
    normalize: bool,
    trust_remote_code: bool,
) -> Callable[[Sequence[str], int, str], np.ndarray]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Hugging Face backend requires transformers. Install with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    except EnvironmentError:
        # Custom loaders (e.g. NomicBert) default to non-safe weights;
        # retry with safe_serialization for repos that only have .safetensors
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            safe_serialization=True,
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

    def embed_texts(texts: Sequence[str], batch_size: int, progress_desc: str, tqdm_pos: int = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # Sort by length to minimize padding waste within batches
        sorted_indices = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in sorted_indices]
        vectors = []
        starts = range(0, len(sorted_texts), batch_size)
        with torch.no_grad():
            for start in tqdm(starts, desc=progress_desc, position=tqdm_pos, leave=False):
                batch_texts = sorted_texts[start : start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = model(**encoded, **extra_forward_kwargs)
                pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        all_vectors = np.concatenate(vectors, axis=0)
        # Restore original order
        unsort = np.empty(len(sorted_indices), dtype=np.intp)
        for new_pos, orig_pos in enumerate(sorted_indices):
            unsort[orig_pos] = new_pos
        return all_vectors[unsort]

    return embed_texts


def build_embedder(
    backend: str,
    model_name: str,
    device: str,
    max_length: int,
    normalize: bool,
    trust_remote_code: bool,
) -> Callable[[Sequence[str], int, str], np.ndarray]:
    if backend not in ("hf", "auto"):
        raise ValueError(f"unknown backend {backend!r} (only 'hf' is supported)")
    return build_hf_embedder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        normalize=normalize,
        trust_remote_code=trust_remote_code,
    )


def collect_code_nodes(graph: nx.Graph):
    node_ids = []
    node_texts = []
    node_types = []
    start_lines = []
    end_lines = []

    for node_id, attrs in graph.nodes(data=True):
        code = attrs.get("code")
        if not isinstance(code, str):
            continue
        code = code.strip()
        if not code:
            continue
        node_ids.append(node_id)
        node_texts.append(code)
        node_types.append(attrs.get("type"))
        start_lines.append(attrs.get("start_line"))
        end_lines.append(attrs.get("end_line"))

    return node_ids, node_texts, node_types, start_lines, end_lines


def _get_repo_name(stem: str) -> str:
    """Extract repo name from graph file stem, e.g. 'django__django-12345' -> 'django__django'."""
    # Find the last '-' followed by digits
    idx = len(stem)
    while idx > 0:
        idx = stem.rfind("-", 0, idx)
        if idx == -1:
            return stem
        if stem[idx + 1:].isdigit():
            return stem[:idx]
    return stem


def _hash_code(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()


def _process_graph(
    graph_file,
    embed_texts,
    args,
    output_dir,
    prev_cache: Optional[Dict[str, Tuple[str, np.ndarray]]] = None,
    gpu_tag="",
    tqdm_pos=None,
):
    """Process a single graph file with incremental embedding.

    prev_cache: dict mapping node_id -> (code_hash, embedding_vector) from a
                previous instance of the same repo. Reuses embeddings for nodes
                whose code hasn't changed.

    Returns (num_nodes, num_embedded, new_cache) where new_cache is the cache
    for the next instance of the same repo.
    """
    stem = Path(graph_file).stem
    out_file = osp.join(output_dir, f"{stem}.node_embeddings.pkl")

    with open(graph_file, "rb") as f:
        graph = pickle.load(f)

    node_ids, node_texts, node_types, start_lines, end_lines = collect_code_nodes(graph)
    num_nodes = len(node_ids)

    if args.dry_run:
        print(f"[dry-run] {gpu_tag}{stem}: {num_nodes} code nodes")
        return num_nodes, 0, None

    if num_nodes == 0:
        payload = {
            "graph_file": graph_file,
            "model_name": args.model_name,
            "backend": args.backend,
            "node_ids": [],
            "node_types": [],
            "start_lines": [],
            "end_lines": [],
            "embeddings": np.zeros((0, 0), dtype=np.float32),
        }
        with open(out_file, "wb") as f:
            pickle.dump(payload, f)
        print(f"[done] {gpu_tag}{stem}: 0 code nodes -> {out_file}")
        return 0, 0, {}

    # Hash current node codes
    code_hashes = [_hash_code(t) for t in node_texts]

    # Determine which nodes need embedding
    if prev_cache:
        need_embed_indices = []
        reused_count = 0
        embed_dim = None
        for i, (nid, h) in enumerate(zip(node_ids, code_hashes)):
            prev = prev_cache.get(nid)
            if prev is not None and prev[0] == h:
                reused_count += 1
                if embed_dim is None:
                    embed_dim = prev[1].shape[0]
            else:
                need_embed_indices.append(i)
    else:
        need_embed_indices = list(range(num_nodes))
        reused_count = 0

    num_to_embed = len(need_embed_indices)

    # Embed only the changed/new nodes
    if num_to_embed > 0:
        texts_to_embed = [node_texts[i] for i in need_embed_indices]
        desc = f"{gpu_tag}{stem} ({num_to_embed}/{num_nodes} new)"
        new_embeddings = embed_texts(texts_to_embed, args.batch_size, desc, tqdm_pos)
    else:
        new_embeddings = None

    # Assemble full embedding matrix
    if reused_count > 0 and prev_cache:
        # Get embed_dim from new embeddings or from cache
        if new_embeddings is not None and new_embeddings.shape[0] > 0:
            embed_dim = new_embeddings.shape[1]
        embeddings = np.empty((num_nodes, embed_dim), dtype=np.float32)
        new_idx = 0
        for i, (nid, h) in enumerate(zip(node_ids, code_hashes)):
            prev = prev_cache.get(nid)
            if prev is not None and prev[0] == h:
                embeddings[i] = prev[1]
            else:
                embeddings[i] = new_embeddings[new_idx]
                new_idx += 1
    else:
        embeddings = new_embeddings

    # Build cache for the next instance
    new_cache = {}
    for i, (nid, h) in enumerate(zip(node_ids, code_hashes)):
        new_cache[nid] = (h, embeddings[i])

    payload = {
        "graph_file": graph_file,
        "model_name": args.model_name,
        "backend": args.backend,
        "node_ids": node_ids,
        "node_types": node_types,
        "start_lines": start_lines,
        "end_lines": end_lines,
        "embeddings": embeddings,
    }

    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"[done] {gpu_tag}{stem}: {num_nodes} nodes, "
        f"{num_to_embed} embedded, {reused_count} reused -> {out_file}"
    )
    return num_nodes, num_to_embed, new_cache


def _gpu_worker(gpu_id, num_gpus, graph_files_for_gpu, args, output_dir):
    """Worker that loads the model on a single GPU and processes its assigned graphs.

    Graph files are sorted so that instances from the same repo are consecutive,
    enabling incremental embedding via the prev_cache.
    """
    device = f"cuda:{gpu_id}"
    gpu_tag = f"[GPU {gpu_id}] "
    # Each GPU gets two tqdm positions: one for the outer graph loop, one for the inner batch loop
    outer_pos = gpu_id * 2
    inner_pos = gpu_id * 2 + 1

    print(f"{gpu_tag}Loading model on {device}, {len(graph_files_for_gpu)} graphs to process")

    embed_texts = build_embedder(
        backend=args.backend,
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
        normalize=args.normalize,
        trust_remote_code=args.trust_remote_code,
    )

    total_nodes = 0
    total_embedded = 0
    # Cache per repo: carries forward embeddings from the previous instance
    repo_cache: Dict[str, Dict[str, Tuple[str, np.ndarray]]] = {}
    pbar = tqdm(
        graph_files_for_gpu,
        desc=f"{gpu_tag}graphs",
        position=outer_pos,
        leave=True,
    )
    for graph_file in pbar:
        stem = Path(graph_file).stem
        repo = _get_repo_name(stem)
        pbar.set_postfix_str(stem, refresh=True)
        prev_cache = repo_cache.get(repo)
        n_nodes, n_embedded, new_cache = _process_graph(
            graph_file, embed_texts, args, output_dir,
            prev_cache=prev_cache, gpu_tag=gpu_tag, tqdm_pos=inner_pos,
        )
        if new_cache is not None:
            repo_cache[repo] = new_cache
        total_nodes += n_nodes
        total_embedded += n_embedded

    print(f"{gpu_tag}Done: {len(graph_files_for_gpu)} graphs, {total_nodes} nodes, {total_embedded} embedded")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for all graph nodes that contain code chunks."
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Directory containing graph pickle files produced by batch_build_graph.py.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Embedding model name/path (for example SWERankEmbed or CodeSAGE model ID).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf"],
        help="Embedding backend. Only Hugging Face transformers is supported.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for embeddings. Defaults to <graph_dir>/node_embeddings/<model_name_slug>/.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_graphs", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for hf backend: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel processing. Each GPU processes a separate graph concurrently.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings (hf backend).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading Hugging Face models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only inspect graph files and count nodes with code, without embedding.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1
    )

    args = parser.parse_args()

    graph_files = list_graph_files(args.graph_dir)
    if args.max_graphs > 0:
        graph_files = graph_files[: args.max_graphs]

    if not graph_files:
        raise FileNotFoundError(f"No .pkl graph files found in {args.graph_dir}")

    model_slug = args.model_name.replace("/", "__")
    output_dir = args.output_dir or osp.join(
        args.graph_dir,
        "node_embeddings",
        model_slug,
    )
    os.makedirs(output_dir, exist_ok=True)

    end_idx = None if args.end_idx < 1 else args.end_idx
    selected_graph_files = graph_files[args.start_idx:end_idx]

    if not selected_graph_files:
        print("No graph files selected after applying start/end indices.")
        return

    pending_graph_files = []
    skipped_count = 0
    for graph_file in selected_graph_files:
        stem = Path(graph_file).stem
        out_file = osp.join(output_dir, f"{stem}.node_embeddings.pkl")
        if osp.exists(out_file) and not args.overwrite:
            print(f"[skip] {stem}: output exists")
            skipped_count += 1
            continue
        pending_graph_files.append(graph_file)

    if not pending_graph_files:
        print(
            f"All selected graphs already processed. skipped={skipped_count}, pending=0"
        )
        return

    print(f"Pending: {len(pending_graph_files)} graphs, skipped: {skipped_count}")
    start_time = time.time()

    num_gpus = args.num_gpus
    if num_gpus > 1 and not args.dry_run:
        import torch
        available = torch.cuda.device_count()
        if num_gpus > available:
            print(f"Warning: requested {num_gpus} GPUs but only {available} available, using {available}")
            num_gpus = available

        # Group by repo so same-repo instances stay on the same GPU for cache reuse,
        # then assign repo groups to GPUs balanced by total count
        repo_groups: Dict[str, List[str]] = defaultdict(list)
        for gf in pending_graph_files:
            repo = _get_repo_name(Path(gf).stem)
            repo_groups[repo].append(gf)
        # Sort each group so instances are processed in order (for incremental cache)
        for repo in repo_groups:
            repo_groups[repo].sort()
        # Assign repo groups to GPUs, greedily balancing load
        gpu_assignments: List[List[str]] = [[] for _ in range(num_gpus)]
        gpu_counts = [0] * num_gpus
        for repo, files in sorted(repo_groups.items(), key=lambda x: -len(x[1])):
            # Assign to the least-loaded GPU
            target = min(range(num_gpus), key=lambda g: gpu_counts[g])
            gpu_assignments[target].extend(files)
            gpu_counts[target] += len(files)

        processes = []
        mp.set_start_method("spawn", force=True)
        for gpu_id in range(num_gpus):
            if not gpu_assignments[gpu_id]:
                continue
            p = mp.Process(
                target=_gpu_worker,
                args=(gpu_id, num_gpus, gpu_assignments[gpu_id], args, output_dir),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"Warning: worker (pid={p.pid}) exited with code {p.exitcode}")
        # Move cursor below all tqdm bars (2 lines per GPU: outer + inner)
        print("\n" * (num_gpus * 2), flush=True)
    else:
        # Single-GPU / dry-run path
        embed_texts = None
        if not args.dry_run:
            embed_texts = build_embedder(
                backend=args.backend,
                model_name=args.model_name,
                device=args.device,
                max_length=args.max_length,
                normalize=args.normalize,
                trust_remote_code=args.trust_remote_code,
            )

        # Sort pending files so same-repo instances are consecutive for cache reuse
        pending_graph_files.sort(key=lambda gf: (_get_repo_name(Path(gf).stem), Path(gf).stem))

        total_nodes = 0
        total_embedded = 0
        repo_cache: Dict[str, Dict[str, Tuple[str, np.ndarray]]] = {}
        for graph_file in pending_graph_files:
            repo = _get_repo_name(Path(graph_file).stem)
            prev_cache = repo_cache.get(repo)
            n_nodes, n_embedded, new_cache = _process_graph(
                graph_file, embed_texts, args, output_dir, prev_cache=prev_cache,
            )
            if new_cache is not None:
                repo_cache[repo] = new_cache
            total_nodes += n_nodes
            total_embedded += n_embedded

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f}s (skipped {skipped_count})")


if __name__ == "__main__":
    main()
