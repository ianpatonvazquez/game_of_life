import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from utils.helpers import compute_mean_std, L1_regularization
from utils.paths import get_stats_paths
from models.mlp import MLP
from data.generate_data import gol_dataset, gol_dataset_rolling
from train.train_model import train_validate
from tests.test_model import testing

# change this parameters to run different tests
num_samples = 2**10  # 1024
n = 15
m = 15
block_size = 1
tau = block_size
rho = 0.5
t0 = 0
dataset_parameters = [num_samples, block_size, m, n, tau, t0, rho]

# see if the mean and std are already computed, if not, compute them
mean_path, std_path = get_stats_paths(block_size, m, n, tau, rho)
if not mean_path.exists() or not std_path.exists():
    print("Mean and/or Std not found, computing them...")
    num_samples = 2**14 # to make the normalization more precise
    dataset_parameters[0] = num_samples  # update num_samples in the parameters
    mean, std = compute_mean_std(dataset_parameters, batch_size=16, verbose=True, normalize=False)
else:
    print(f"Mean and Std already computed")

# create the model
input_size = (m//block_size) ** 2 * tau
hidden_size = [input_size//2, input_size//4]
output_size = 1
model = MLP(input_size, hidden_size, output_size, dropout_rate=0.1)
print(f"Model created with input size {input_size}, hidden sizes {hidden_size}, and output size {output_size}")

# train the model
batch_size = 16
train_size = 16*1000
val_size = 2**10
dataset_parameters[0] = val_size
trained_model, train_losses, val_losses, acc_batch = train_validate(model, batch_size, train_size, val_size, dataset_parameters, fixed=True, conv=False, balance=True, lambda_l1=0.01, printing_rhythm = 10, verbose=True)

# test the model
test_size = 2**10
dataset_parameters[0] = test_size
accuracy, loss = testing(trained_model, dataset_parameters, batch_size=16, verbose=True)
