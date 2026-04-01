"""Reproducibility: set all random seeds."""
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover (optional dependency)
    torch = None


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
