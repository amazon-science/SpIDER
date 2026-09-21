# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters and configuration for the SpIDER localization pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def parse_edge_depths(spec: str) -> Dict[str, int]:
    """Parse "contains:4,invokes:2" -> {"contains": 4, "invokes": 2}.

    Accepts ',' or ';' separators. Empty/blank spec -> {}. Raises ValueError on a
    malformed token so a typo fails loudly instead of silently dropping an override.
    """
    out: Dict[str, int] = {}
    for pair in (spec or "").replace(";", ",").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"bad token {pair!r}, expected 'edge:depth' (e.g. 'invokes:2')")
        etype, d = pair.split(":", 1)
        out[etype.strip()] = int(d.strip())
    return out


@dataclass
class SelectorConfig:
    # --- Data paths ---
    # scores_path : issue-node score pickle (see data_prep/, or the README's
    #               "Score pickle schema" section).
    # graphs_dir  : directory of per-instance repository graphs, {instance_id}.pkl.
    scores_path: str = ""
    graphs_dir: str = ""

    # --- Run-level settings ---
    node_type: str = "function"          # "file", "class", or "function"
    ranking_type: str = "dense"          # "dense" or "bm25" -- key into the score pickle
    edge_type: str = "contains"          # "contains", "imports", "inherits", "invokes"
    bfs_depth: int = 4

    # Multi-edge graph exploration. `edges` (comma/space separated) supersedes
    # `edge_type` when set; each edge expands independently from the seed at its
    # own depth, so mixed-type paths are not traversed. `bfs_depth_per_edge`
    # overrides `bfs_depth` per edge -- important for dense edges like `invokes`,
    # where depth 4 can blow up the frontier by orders of magnitude.
    edges: Optional[str] = None                                      # e.g. "contains,invokes"
    bfs_depth_per_edge: Dict[str, int] = field(default_factory=dict)  # e.g. {"invokes": 2}

    # --- Hyperparameters ---
    K: int = 20        # nodes per ranked window
    L: int = 20        # final selection budget
    N: int = 500       # neighbor candidate pool (ranks K+1..K+N)
    C: int = 5         # centers explored per round

    # --- Context control ---
    max_code_chars: int = 1500           # truncate code chunks beyond this
    max_steps: int = 1                   # rounds of the SpIDER loop
    # Cap on BFS neighbors shown per center (0 = unlimited). Neighbors are kept
    # in rank order, so this drops the lowest-ranked. Needed for dense edges like
    # `invokes`, whose BFS can cover the whole N-window and overflow the context.
    max_neighbors_per_center: int = 0

    # --- LLM provider: "azure_responses", "azure", "openai", or "vllm" ---
    provider: str = "openai"
    model: str = "gpt-4o"                # model name / deployment name
    temperature: float = 0.0
    max_tokens: int = 8192               # max LLM response tokens

    # --- Generic OpenAI-compatible (provider="openai") ---
    api_base_url: Optional[str] = None        # e.g. "https://X.openai.azure.com/openai/v1"
    api_key: Optional[str] = None

    # --- Azure native SDK (provider="azure") ---
    azure_api_base: Optional[str] = None      # e.g. "https://<resource>.cognitiveservices.azure.com"
    azure_api_version: Optional[str] = None   # e.g. "2025-04-01-preview"
    azure_api_key: Optional[str] = None
    azure_deployment: Optional[str] = None    # deployment name

    # --- vLLM / local OpenAI-compatible server (provider="vllm") ---
    vllm_base_url: Optional[str] = None       # e.g. "http://localhost:8000/v1"
    vllm_api_key: Optional[str] = None        # usually "EMPTY" for local

    # --- Output ---
    output_dir: str = ""

    def ranking_key(self) -> str:
        """Key into issue data for the ranked list."""
        return "dense_ranked" if self.ranking_type == "dense" else "bm25_ranked"

    def edge_list(self) -> List[str]:
        """Edge types to explore. Falls back to the single `edge_type`."""
        if self.edges:
            return [e for e in self.edges.replace(",", " ").split() if e]
        return [self.edge_type]

    def edge_depth_map(self) -> Dict[str, int]:
        """{edge_type -> BFS depth}, per-edge override winning over `bfs_depth`."""
        return {e: int(self.bfs_depth_per_edge.get(e, self.bfs_depth))
                for e in self.edge_list()}

    def edge_slug(self) -> str:
        """Filename-safe descriptor of the explored edges and their depths.

        e.g. "contains4", "invokes2", "contains4+invokes2".
        """
        m = self.edge_depth_map()
        return "+".join(f"{e}{m[e]}" for e in sorted(m))

    def run_slug(self) -> str:
        """Descriptor of the graph-exploration settings, for dir and file names.

        e.g. "edges-contains4", "edges-contains4+invokes2_nbrcap200". Single source
        of truth for both the results filename tag and the output subdirectory, so
        two edge configs can never write to the same path.
        """
        s = f"edges-{self.edge_slug()}"
        if self.max_neighbors_per_center:
            s += f"_nbrcap{self.max_neighbors_per_center}"
        return s
