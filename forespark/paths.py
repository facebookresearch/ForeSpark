# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Locations of bulky artifacts, from ``configs/paths.yaml``.

python -m forespark.paths     # show where everything points
"""

import os
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_CONFIG = yaml.safe_load((ROOT / "configs/paths.yaml").read_text()) or {}


def resolve(key: str, default: Path | str) -> Path:
    """A configured path, or ``default`` if the key is blank; ``FORESPARK_<KEY>``
    in the environment overrides both (a smoke test's scratch data, say).

    Relative values are anchored at the repo root, so a run started from
    anywhere writes to the same place.
    """
    value = os.environ.get(f"FORESPARK_{key.upper()}") or _CONFIG.get(key) or default
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else ROOT / path


DATA = resolve("data", "data")
RUNS = resolve("runs", "runs")
VENV = resolve("venv", ".venv")
RESULTS = resolve("results", "results")
BENCH_DATA = DATA / "bench"
HIDDEN_STATES = Path(
    os.environ.get("HIDDEN_STATES_PATH")
    or resolve("hidden_states", Path(tempfile.gettempdir()) / "hidden_states")
)


if __name__ == "__main__":
    for name, path in sorted(globals().items()):
        if name.isupper() and isinstance(path, Path) and name != "ROOT":
            note = "" if path.exists() else "   (does not exist yet)"
            print(f"{name.lower():<14} {path}{note}")
