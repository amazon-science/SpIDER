# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal git-patch parser: which lines does a patch touch, in which file?

Extracted from LocAgent (github.com/gersteinlab/LocAgent) and reduced to the one
function the ground-truth builder needs. Language-agnostic on purpose --
SWE-PolyBench patches touch .java/.js/.ts files, not just .py.
"""

from collections import defaultdict

import unidiff


def parse_patch(patch: str):
    """Parse a git patch into [{file, hunks: [{start_line, changes}]}].

    `changes` maps "add"/"delete" to [{content, line}], where `line` is the line
    number in the target (for additions) or source (for deletions) revision.
    Whitespace-only changes are skipped -- they never identify a code entity.
    """
    parsed_patches = []
    patch_set = unidiff.PatchSet(patch)

    for patched_file in patch_set:
        parsed_file_patch = {"file": patched_file.path, "hunks": []}

        for hunk in patched_file:
            parsed_hunk = {
                "start_line": hunk.source_start,
                "changes": defaultdict(list),
            }
            for line in hunk:
                if not str(line)[1:].strip():
                    continue
                if line.is_removed:
                    parsed_hunk["changes"]["delete"].append({
                        "content": str(line)[1:],
                        "line": line.source_line_no,
                    })
                elif line.is_added:
                    parsed_hunk["changes"]["add"].append({
                        "content": str(line)[1:],
                        "line": line.target_line_no,
                    })
            parsed_file_patch["hunks"].append(parsed_hunk)

        parsed_patches.append(parsed_file_patch)

    return parsed_patches
