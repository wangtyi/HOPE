import random
import os
import numpy as np
import torch


def seed_everything(seed=123):
    """
    Set random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False