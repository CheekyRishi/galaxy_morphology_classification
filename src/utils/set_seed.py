import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for full reproducibility across Python, NumPy, and PyTorch.

    Note:
    - Enables deterministic behavior (slower but reproducible)
    - Disable cudnn benchmark to avoid nondeterministic kernels
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
