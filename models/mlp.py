import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F

# MLP: Multi Layer Perceptron
# This is a fully connected neural network with multiple hidden layers
class MLP(nn.Module):

  def __init__(self, input_size, hidden_size, output_size, dropout_rate):
    """
    Initialization of the MLP

        Parameters: num_input, num_output, num_hidden
        - input_size: number of input neurons
        - output_size: number of output neurons
        - hidden_size: number of neurons in hidden layers (list because the number can differ layer to layer)

    """

    # calls the constructor of nn.Module (needed to use NN functionaliaties)
    super(MLP, self).__init__()

    # array that will be the layers
    layers = []

    # number of neurons that each layer will have (this variable will be updated each time)
    num_neurons = input_size

    # loop that creates the layers of the required size
    for num_hidden in hidden_size:

      # nn.Linear creates a fully connected layer
      layers.append(nn.Linear(num_neurons, num_hidden))

      # batch normalization
      layers.append(nn.BatchNorm1d(num_hidden))

      # activation function - ReLU: Rectified Linear Unit (f(x) = max(0,x)): used in hidden layers
      layers.append(nn.ReLU())

      # apply the Dropout
      layers.append(nn.Dropout(p=dropout_rate))

      # update the number of neurons
      num_neurons = num_hidden

    # we connect the last two matrices
    layers.append(nn.Linear(num_neurons, output_size))

    # sigmoid activation because of the binary outcome
    #layers.append(nn.Sigmoid())

    # complete model of the neural network: layers + activation functions: *unpacks the list
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    """We run an input tensor (x) through the model we defined before"""
    x = x.view(x.size(0),-1)
    # we return the input tensor but with our mlp applied
    return self.model(x)