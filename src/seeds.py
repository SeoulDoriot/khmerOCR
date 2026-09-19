"""Reproducibility: one place that fixes every random source.

The rubric requires seeds fixed across Python, NumPy and PyTorch. Import and
call set_all_seeds() at the top of every experiment so results are repeatable
and the comparison between approaches is not luck.
"""
import os
import random

import numpy as np
import torch

SEED = 42


def set_all_seeds(seed: int = SEED, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # slower, but removes cuDNN autotuning nondeterminism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
