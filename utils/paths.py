from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NORMALIZATION_DIR = BASE_DIR / "normalization_data"

def get_stats_paths(block_size, m, n, tau, rho):
    name = f"{block_size}_{m}_{n}_{tau}_{rho}"
    mean_path = NORMALIZATION_DIR / f"{name}_mean.npy"
    std_path = NORMALIZATION_DIR / f"{name}_std.npy"
    return mean_path, std_path