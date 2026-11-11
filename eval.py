#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmarks: ./eval.py --help (implementation in forespark/bench)."""

import sys

from forespark.bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
