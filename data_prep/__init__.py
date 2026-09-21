# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data preparation for SpIDER: embeddings -> issue-node scores -> BFS pool.

Three stages, each a standalone `python -m data_prep.<module>` entry point:

  1. embed_graph_nodes          embed every code-bearing graph node
  2. compute_issue_node_scores  score each issue against its repo's nodes
  3. add_filtered_bfs_rankings  attach the BFS-expanded top-500 candidate pool

See ../README.md for the full command sequence and the output schema.
"""
