import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from data.generate_data import gol_dataset_rolling as gol_dataset
from models.mlp import MLP
from utils.helpers import L1_regularization


# TEST THE MODEL GENERATING THE DATASET INSIDE
def testing(model, batch_size, dataset_parameters, fixed, conv, verbose=True, balance=False):

  """

  Function that tests the accuracy of the neural network

  """

  # create the training data and load it
  num_samples, block_size, m, n, tau, t0, rho = dataset_parameters
  test_dataset = gol_dataset(num_samples, block_size, m, n, tau, t0, rho, balance, normalize=True)

  # load the data and test everything
  test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False) #skip it
  loss_fn = nn.BCEWithLogitsLoss()

  # first we have to turn into evaluation mode with this command
  model.eval()

  # for faster testing, disable gradient computation
  with torch.no_grad():

    # monitorizing loss and correct predictions
    correct = 0
    total = 0
    zero_count = 0
    one_count = 0
    total_loss = 0
    real_one = 0
    real_zero = 0
    i = 1

    # loop that goes through the batches of data
    for batch in test_loader:

      # input data (x) and labels (y)
      x, y_label = batch

      # we need to flatten the input to match the dimensions when training
      x = x.view(x.size(0), -1)

      # output y_out obtained with our trained model
      y_out = model(x).squeeze()

      # as we get a P(0,1) we need to convert to binary values
      y_pred = (y_out > 0.5).float()

      # compute loss and count it
      loss = loss_fn(y_out, y_label)
      total_loss += loss.item()

      # count correct predictions
      correct += (y_pred == y_label).float().sum().item()

      # count the amount of zeros and ones
      zero_count += (y_pred == 0).float().sum().item()
      one_count += (y_pred == 1).float().sum().item()
      real_one += (y_label == 1).float().sum().item()
      real_zero += (y_label == 0).float().sum().item()

      print(f"Batch: {i}, Prediction: {y_pred}, Real: {y_label}")

      # count total predictions
      total += y_label.size(0)

      i+=1
  # compute accuracy and average loss
  accuracy = correct / total
  avg_loss = total_loss / len(test_loader)

  # print the results
  if verbose:
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Predicted one counter: {int(one_count)}, Predicted zero counter: {int(zero_count)}")
    print(f"Real one counter: {int(real_one)}, Real zero counter: {int(real_zero)}")

  # return accuracy and loss for the dataset
  return accuracy, avg_loss