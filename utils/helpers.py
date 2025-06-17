import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from data.generate_data import gol_dataset_rolling
from utils.paths import get_stats_paths


# MEAN & STD COMPUTER
def compute_mean_std(dataset_parameters, batch_size=16, verbose=True, normalize=False):
  """
  Function that computes the mean and standard deviation of the dataset

  """

  # create the training data and load it
  num_samples, block_size, m, n, tau, t0, rho = dataset_parameters

  # load the dataset and dataloader
  dataset = gol_dataset_rolling(num_samples, block_size, m, n, tau, t0, rho, normalize=False)

  # turn into np
  data = np.array(dataset.data)

  # compute the mean and standard deviation
  mean = data.mean(axis=0)
  std = data.std(axis=0)

  # get paths using helper function
  mean_path, std_path = get_stats_paths(block_size, m, n, tau, rho)
  file_path = "/content/drive/My Drive/Colab Notebooks/"
  np.save(mean_path, mean)
  np.save(std_path, std)

  # print the results
  if verbose:
    print(f"Mean: {mean}")
    print(f"Mean saved to: {str(mean_path)}")
    print(f"Std: {std}")
    print(f"Std saved to: {str(std_path)}")


  return mean, std

# L1 REGULARIZATION

def L1_regularization(model, lambda_l1):
  """
  Function that implements L1 Regularization:
    it adds a penalty to the loss function based on the abs(weights) to encourage sparsity and reduce overfitting

  Parameters:
  - model: neural network model
  - lambda_l1: parameter that controls the weight of the penalty term

  """

  # initialize the loss
  L1_loss = 0

  # add the loss corresponding to the absolute values
  for param in model.parameters():
    L1_loss += torch.sum(torch.abs(param))

  # return the loss with the weight controlling parameter
  return lambda_l1 * L1_loss

def set_all_seeds(seed):
    # numpy seed
    np.random.seed(seed)
    # torch seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
