import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from data.generate_data import gol_dataset_rolling
from models.mlp import MLP
from utils.helpers import L1_regularization

# TRAINING AND VALIDATING
def train_validate(model, batch_size, train_size, val_size, dataset_parameters, fixed, conv, balance, lambda_l1=0.01, printing_rhythm=10, verbose=True):

  """
  Function that trains the model and validates it

  Parameters:
  - model: MLP neural network model
  - batch_size: size of the batches
  - train_size: size of the training data
  - val_size: size of the validation data
  - fixed: if True, fixed grid dimensions
          if False, multipliers
  - conv: if True, convolutional coarse graining
          if False, block average coarse graining
  - balance: if True, balance the dataset (equal number of alive and dead cells)
  - lambda_l1: parameter that controls the weight of the penalty term
  - printing_rhythm: how often to print the loss
  - dataset_parameters: parameters of the dataset (num_samples, block_size, m_multiplier, n_multiplier, tau_multiplier, t0, rho)
  - verbose: if True, prints the loss

  Returns:
  - training_loss
  - validation_loss
  - accuracy

  """

  # initialize the losses
  train_losses_per_batch = []
  val_losses_per_batch = []
  accuracy_per_batch = []

  # create the training data and load it
  num_samples, block_size, m, n, tau, t0, rho = dataset_parameters
  training_dataset = gol_dataset_rolling(train_size, block_size, m, n, tau, t0, rho, normalize=True)
  train_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle=True)

  # now, we generate validation data
  validation_dataset = gol_dataset_rolling(val_size, block_size, m, n, tau, t0, rho, normalize=True)
  validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)

  # print the length of the dataset
  print(f"Length of the training dataset: {len(training_dataset)}")

  # optimizer and loss function
  optimizer = optim.Adam(model.parameters(), lr = 1e-3)
  loss_fn = nn.BCEWithLogitsLoss()
  # loss_fn = nn.BCELoss()

  # different optimizer?
  # test different gridsizes

  # loop that goes through the batches of the training data
  for i, batch in enumerate(train_loader):

    # evaluation before training
    with torch.no_grad():
        model.eval()
        correct_before = 0
        total_before = 0
        for x_full, y_full in train_loader:
            x_full = x_full.view(x_full.size(0), -1)
            y_pred_full = model(x_full).squeeze()
            y_pred_binary_full = (y_pred_full > 0.5).float()
            correct_before += (y_pred_binary_full == y_full).float().sum().item()
            total_before += y_full.size(0)
        accuracy_before = correct_before / total_before

    # print only after printing rhythm
    if verbose and i % printing_rhythm == 0:
      print(f"[Pre-batch {i}] Accuracy on full training data: {accuracy_before:.4f}")

    # initialize the counters
    model.train()
    correct_train = 0
    total_train = 0
    zero_count_train = 0
    one_count_train = 0

    # establish training
    model.train()

    # define the x,y
    x_training, y_training = batch

    # input data
    x_training = x_training.view(x_training.size(0), -1)

    # reset gradients
    optimizer.zero_grad()

    # predict with the model (forward pass) (.squeeze erases the extradimension)
    y_pred = model(x_training).squeeze()

    # compute the training loss
    loss = loss_fn(y_pred, y_training)
    raw_loss = loss.item()

    # add L1 regularization
    loss += L1_regularization(model, lambda_l1)

    # backpropagation: compute gradient for the weights
    loss.backward()

    # to prevent exploding gradients (try vanishing gradients too)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    # update weights with the optimizer (Adam in our case)
    optimizer.step()

    # save the data lost by batch
    train_losses_per_batch.append(raw_loss)

    # as we get a P(0,1) we need to convert to binary values
    y_acc_train = (y_pred > 0.5).float()

    # count correct predictions
    correct_train += (y_acc_train == y_training).float().sum().item()

    # count the amount of zeros and ones
    zero_count_train += (y_acc_train == 0).float().sum().item()
    one_count_train += (y_acc_train == 1).float().sum().item()

    # count total predictions
    total_train += y_training.size(0)

    # accuracy training
    accuracy_train = correct_train / total_train

    # define the validation loss
    val_loss = 0

    # disable the gradient
    with torch.no_grad():

      correct = 0
      total = 0
      zero_count = 0
      one_count = 0
      model.eval()

      # loop that goes through the batches of data in the validation data
      for batch1 in validation_loader:

        # batch
        x_val, y_val = batch1

        # prepare input
        x_val = x_val.view(x_val.size(0), -1)

        # predict the y
        y_pred = model(x_val).squeeze()

        # as we get a P(0,1) we need to convert to binary values
        y_acc = (y_pred > 0.5).float()

        # count correct predictions
        correct += (y_acc == y_val).float().sum().item()

        # count the amount of zeros and ones
        zero_count += (y_acc == 0).float().sum().item()
        one_count += (y_acc == 1).float().sum().item()

        # count total predictions
        total += y_val.size(0)

        # compute the loss
        loss_val = loss_fn(y_pred, y_val)

        # add the loss in each step
        val_loss += loss_val.item()

      # average the validation loss for the data loaded and add it to the returned data
      val_loss /= len(validation_loader)
      val_losses_per_batch.append(val_loss)

      # compute accuracy and average loss
      accuracy = correct / total
      accuracy_per_batch.append(accuracy)

      # add convergence when it learns
      #if i >= 4 and (accuracy_per_batch[i] == accuracy_per_batch[i-1] == accuracy_per_batch[i-2] == accuracy_per_batch[i-3] == accuracy_per_batch[i-4] == 1.):
      #  print("Convergence reached: Model trained")
      #  break


      # print the values obtained if we want
      if verbose and i % printing_rhythm == 0:
        print(f"Batch: {i}, Training Loss: {raw_loss:.6f}, Validation Loss: {val_loss:.6f}, TRAINING: One counter: {int(one_count_train)}, Zero counter: {int(zero_count_train)}, Accuracy: {accuracy_train} \| VALIDATION: One counter: {int(one_count)}, Zero counter: {int(zero_count)}, Accuracy: {accuracy}")

  return model, train_losses_per_batch, val_losses_per_batch, accuracy_per_batch
