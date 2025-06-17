import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F

# CNN: Convolutional Neural Network
# This is a simple CNN with two convolutional layers
# The first layer has 2 filters and the second layer has 1 filter
class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    # first layer with 2 filters
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=0, bias = True)

    # These weights and biases are the ones used in the article by Bibin and Dereventsev and they achieve perfect accuracy
    #self.conv1.weight.data = torch.tensor([
    #    [[[1,1,1],
    #      [1, 3/5, 1],
    #      [1, 1, 1]]],
    #    [[[1,1,1],
    #      [1, 2/5, 1],
    #      [1,1,1]]]
    #], dtype=torch.float32)
    #self.conv1.bias.data = torch.tensor([-2.4, -3.6], dtype=torch.float32)

    # second layer with 1 filter
    self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=True)
    #self.conv2.weight.data = torch.tensor([[[[2]], [[-2]]]], dtype=torch.float32)
    #self.conv2.bias.data = torch.tensor([-1], dtype=torch.float32)

  def step(self, x, n):
    """
    Single step of the CNN
    """
    m = x.shape[-1]

    # circular padding when n = m
    if n == m:
      x = F.pad(x, pad=(1,1,1,1), mode="circular")

    # padding with zeros when n > m
    else:

      # change the value = 0 to 0.5 (try both)
      x = F.pad(x, pad=(1,1,1,1), mode="constant", value=0.5)

    # step 1: conv + tanh
    x = self.conv1(x)
    x = torch.tanh(x)

    # step 2: conv + relu
    x = self.conv2(x)
    x = F.relu(x)

    return x

  def forward(self, x, steps, n):
    """
    Recursivity function
    """
    for _ in range(steps):
      x = self.step(x, n)

    return x