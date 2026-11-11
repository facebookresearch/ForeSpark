#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train a drafter against a running hidden-state server.

    ./serve.py qwen14 --gpus 0,1              # hidden states
    ./train.py qwen14 forespark --gpus 2,3,4,5,6,7

Methods: forespark (DFlash2 with confidence and width-conditioned Fore-D heads,
lr 6e-4) and dspark (the baseline, lr 3e-4), both with non-causal draft
attention. Unknown flags pass through to speculators' train.py.
"""

import sys

from forespark.train.launch import main

if __name__ == "__main__":
    sys.exit(main())
