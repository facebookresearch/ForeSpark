# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read Fore-D's step log (`eval.py ... --drafter-width d --fore-d-log DIR`: forced widths, all verified).

Each logged draft is paired with its outcome: the accepted count arrives with
the same request's next step (num_sampled - 1). Drafts whose request did not
come back (finished, preempted) or that could not draft (legal width 0) drop.
"""

from pathlib import Path

import numpy as np


def load_drafts(directory: Path) -> dict[str, np.ndarray]:
    """Arrays over drafts: step, batch, req, fore_d_logits, confidence, legal, accepted."""
    chunks = [np.load(path) for path in sorted(directory.glob("steps.*.npz"))]
    if not chunks:
        raise SystemExit(f"no steps.*.npz under {directory}")
    data = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0].files}
    outcome = {
        (int(s), r): int(a) - 1
        for s, r, a in zip(data["step"], data["req"], data["num_sampled"], strict=True)
    }
    keep, accepted = [], []
    for i, (s, r) in enumerate(zip(data["step"], data["req"], strict=True)):
        a = outcome.get((int(s) + 1, r))
        if a is not None and data["legal"][i] > 0:
            keep.append(i)
            accepted.append(min(a, int(data["legal"][i])))
    keep = np.asarray(keep)
    drafts = {
        key: data[key][keep]
        for key in ("step", "batch", "req", "fore_d_logits", "confidence", "legal")
    }
    drafts["accepted"] = np.asarray(accepted)
    return drafts
