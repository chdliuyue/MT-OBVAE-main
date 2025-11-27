import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int] = None) -> None:
    """Seed Python, NumPy and (optionally) PyTorch RNGs.

    The helper is resilient to PyTorch being absent so that classical models can
    still call it without introducing additional dependencies.
    """

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # PyTorch is optional; silently skip if it isn't installed.
        pass
