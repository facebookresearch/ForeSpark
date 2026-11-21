# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from forespark.bench.datasets import load_prompts


def test_load_prompts_closes_file_after_reading(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "first"}\n\n  \n{"prompt": "second"}\n')

    with path.open() as file, patch.object(Path, "open", return_value=file):
        rows = [{"prompt": "first"}, {"prompt": "second"}]
        assert load_prompts(path) == rows
        assert file.closed


def test_load_prompts_closes_file_after_invalid_json(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "first"}\ninvalid json\n')

    with path.open() as file, patch.object(Path, "open", return_value=file):
        with pytest.raises(json.JSONDecodeError):
            load_prompts(path)
        assert file.closed
