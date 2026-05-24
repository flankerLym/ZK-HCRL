from __future__ import annotations
import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))
