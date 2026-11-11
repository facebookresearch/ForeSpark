#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fit the Fore-D and confidence heads on a frozen drafter (forespark/train/frozen_heads.py).

    ./serve.py glm53flash --gpus 0,1,2,3                                   # target hidden states
    ./heads.py extract glm53flash CKPT --split train --batches 0:16000 --out X/fit
    ./heads.py extract glm53flash CKPT --split val --batches 0:500 --out X/eval
    ./heads.py fit glm53flash CKPT X/fit --eval X/eval --out X/heads
    ./heads.py merge CKPT X/heads/heads-fit@all.safetensors --out X/checkpoint

extract saves the heads' inputs and labels from packed corpus batches; fit
trains both heads on them and reports them beside the checkpoint's own heads;
merge writes the drafter with the fitted heads. scripts/training/heads.sh runs
all of it.
"""

import sys

from forespark.train.frozen_heads import main

if __name__ == "__main__":
    sys.exit(main())
