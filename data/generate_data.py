import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
import scipy.signal
from utils.paths import get_stats_paths

def computing_neighbour_sum(grid):
  """
  Computes the neighbours' sum for each cell in the grid with periodic boundary conditions
  """
  return (
    np.roll(np.roll(grid, 1, axis=0), 1, axis=1)  # top-left neighbor
    + np.roll(np.roll(grid, 1, axis=0), 0, axis=1)  # top neighbor
    + np.roll(np.roll(grid, 1, axis=0), -1, axis=1)  # top-right neighbor
    + np.roll(np.roll(grid, 0, axis=0), 1, axis=1)  # left neighbor
    + np.roll(np.roll(grid, 0, axis=0), -1, axis=1)  # right neighbor
    + np.roll(np.roll(grid, -1, axis=0), 1, axis=1)  # bottom-left neighbor
    + np.roll(np.roll(grid, -1, axis=0), 0, axis=1)  # bottom neighbor
    + np.roll(np.roll(grid, -1, axis=0), -1, axis=1)  # bottom-right neighbor
    )

# GAME OF LIFE NEXT STEP GENERATOR
def next_step_fn(grid):
    """
    Function that computes the next grid in Conway's Game of Life using periodic boundary conditions
    """

    # we are shifting the grid while wrapping around at the edges (torus) so as to impose the periodic boundary conditions
    neighbours_sum = computing_neighbour_sum(grid)

    # grid's length as a variable
    n = len(grid)

    # Conway's Game of Life rules
    for i in range(n):
        for j in range(n):

            # Cells with alive elements
            if grid[i][j] == 1:
                if (neighbours_sum[i][j] <= 1) or (neighbours_sum[i][j] >= 4):
                    grid[i][j] = 0

            # Cells with dead elements
            elif grid[i][j] == 0:
                if (neighbours_sum[i][j] == 3):
                    grid[i][j] = 1

    # we return the grid
    return grid

# COARSE GRAINING IMPLEMENTATION
def block_average(grid, block_size):
    """
    Function that applies coarse-graining for a block_size x block_size

    """
    # length parameters
    n = len(grid)

    # we compute how many blocks will there be in the coarsed grid
    new_blocks = n//block_size

    # initialize sum variables
    alive_sum = 0
    dead_sum = 0

    # initialize coarsed grid
    coarse_grid = np.zeros((new_blocks,new_blocks),dtype=int)

    # loops to go through the future blocks
    for i in range(new_blocks):

        for j in range(new_blocks):

            # we create a subgrid by slicing our original grid
            subgrid = grid[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

            # wrap when the dimensions do not match
            # x-axis
            if subgrid.shape[0] < block_size:
              subgrid = np.vstack([subgrid, grid[0:block_size - subgrid.shape[0], j * block_size:(j + 1) * block_size]])
            # y-axis
            if subgrid.shape[1] < block_size:
              subgrid = np.hstack([subgrid, grid[i * block_size:(i + 1) * block_size, 0:block_size - subgrid.shape[1]]])

            # we sum over the elements of the subgrid
            alive_sum = np.sum(subgrid)
            dead_sum = block_size**2 - alive_sum

            # let's see if we have more alive or dead and coarse-grain as such
            if alive_sum > dead_sum:
                coarse_grid[i,j] = 1
            else:
                coarse_grid[i,j] = 0

    return coarse_grid

def convolution_coarse_graining(grid, block_size):
    """
    Applies a convolution-like coarse-graining operation without reducing dimensions.
    Each cell is replaced by the majority value in its local block_size x block_size neighborhood.
    """

    # kernel sum (best one for uniform)
    kernel = np.ones((block_size, block_size))

    # convolution operation
    conv_result = scipy.signal.convolve2d(grid, kernel, mode='same', boundary='wrap')

    # threshold
    threshold = (block_size**2) / 2

    # apply majority rule
    coarse_grid = (conv_result > threshold).astype(int)

    return coarse_grid

# GAME OF LIFE DATASET IMPLEMENTATION
# This dataset generates sequences of grids according to the Game of Life rules.
class gol_dataset():
  def __init__(self, num_samples, block_size, m, n, tau, t0, rho, balance, normalize):

    """Initialization of the dataset: converting data into PyTorch tensors

    Parameters: num_samples, n, m, tau, block_size
    - num_samples: number of samples (random grids) that will be created
    - block_size: size of the coarsed grid (optional)
    - m: dimensions of the subgrid
    - n: dimensions of the grid
    - tau: how many generations will we generate?
    - t0: initialization time
    - rho: density
    - balance: if True, balance the dataset (equal number of alive and dead cells)
    Data: grid sequences over a tau generations
    Labels: center cell of the tau+1 generation

    """

    # number of samples
    self.num_samples = num_samples

    # grid coarsing parameter
    self.block_size = block_size

    # fixed parameters
    self.n = n
    self.m = m
    self.tau = tau

    # initialization time
    self.t0 = t0

    # density
    self.rho = rho

    # important to check whether the n and m values are possible
    assert n >= m, "n must be greater or equal than m"

    # data initialization
    self.data = []

    # labels initialization
    self.labels = []

    # generate the amount of data we want with balance
    if balance:
      target = num_samples//2
      alive_count = 0
      dead_count = 0
    else:
      total_target = num_samples

    counter = 0
    while True:

      # to add the data to the tensors we need an index counter = index
      index = 0

      if balance:
        if alive_count >= target and dead_count >= target:
          break
      else:
        if len(self.data) >= total_target:
          break

      # NxN grid
      grid = np.random.choice([0,1],(n,n),p=[1 - rho, rho])

      # we select the top-left corner from which we'll start the MxM subgrid
      i, j = np.random.randint(0, n-m+1, 2)

      # initialize sequence: matrix that will store tau grids
      sequence = np.zeros((tau, m//block_size, m//block_size), dtype=np.float32)

      # loop that iterates the initial grid t_0 times to arrive to the initialization time
      for t in range(t0):

        # we update the grid t0 generations
        grid = next_step_fn(grid)

      # loop that iterates the initial grid tau times according to the game of life rules
      for t in range(tau):

        # choose an MxM subgrid
        subgrid = grid[i:i+m, j:j+m]

        # apply coarse-graining
        coarse_grid = block_average(subgrid, block_size)

        # add to the data
        sequence[t] = coarse_grid

        # update the grid
        grid = next_step_fn(grid)

      # coarse grain the tau+1 grid
      subgrid = grid[i:i+m, j:j+m]
      coarse_grid = block_average(subgrid, block_size)
      label = coarse_grid[m//(2*block_size), m//(2*block_size)]

      if balance:
        if int(label) == 1 and alive_count < target:
          self.data.append(sequence)
          self.labels.append(label)
          alive_count += 1
          index += 1
        elif int(label) == 0 and dead_count < target:
          self.data.append(sequence)
          self.labels.append(label)
          dead_count += 1
          index += 1
      else:
        self.data.append(sequence)
        self.labels.append(label)

      # have 2 datasets: don't filter one (balance and imbalanced)

      counter += 1
    print(f"Counter : {counter}, Dead count : {dead_count}, Alive coun : {alive_count}")

    if normalize:
      self.normalize()

    # normalize data
    # turn into torch tensors
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.labels = torch.tensor(self.labels, dtype=torch.float32)

  def __len__(self):
    """Length of the dataset (how many samples exist)"""
    length = len(self.labels)
    return length

  def __getitem__(self,idx):
    """Retrieve a single sample from the dataset"""

    # data as the x vector
    x = self.data[idx]

    # labels as the y vector
    y = self.labels[idx]

    return x,y

  def normalize(self):
    """Normalize dataset using precomputed mean and std"""

    self.data = np.array(self.data)
    mean_path, std_path = get_stats_paths(self.block_size, self.m, self.n, self.tau, self.rho)
    try:
      mean = np.load(mean_path)
      std = np.load(std_path)
    except:
      raise RuntimeError("Mean and std files not found")

    # avoid NaN
    eps = 1e-6
    std[std < eps] = 1
    self.data = (self.data - mean) / std

# ROLLING DATASET IMPLEMENTATION
# This dataset is similar to the gol_dataset, but it does not wait for the center cell to be zero.
# Instead, it looks for zeros in the subgrid and rolls the sequence accordingly.
class gol_dataset_rolling():
  def __init__(self, num_samples, block_size, m, n, tau, t0, rho, normalize):

    """
    Initialization of the dataset: converting data into PyTorch tensors (rolling version)
    Rolling implies that we will look for zeros when balancing the dataset, not waiting for the center cell to be zero.

    Parameters: num_samples, n, m, tau, block_size
    - num_samples: number of samples (random grids) that will be created
    - block_size: size of the coarsed grid (optional)
    - m: dimensions of the subgrid
    - n: dimensions of the grid
    - tau: how many generations will we generate?
    - t0: initialization time
    - rho: density
    - balance: if True, balance the dataset (equal number of alive and dead cells)
    Data: grid sequences over a tau generations
    Labels: center cell of the tau+1 generation

    """

    # number of samples
    self.num_samples = num_samples

    # grid coarsing parameter
    self.block_size = block_size

    # fixed parameters
    self.n = n
    self.m = m
    self.tau = tau

    # initialization time
    self.t0 = t0

    # density
    self.rho = rho

    # important to check whether the n and m values are possible
    assert n >= m, "n must be greater or equal than m"

    # data initialization
    self.data = []

    # labels initialization
    self.labels = []

    # generate the amount of data we want with balance
    target = num_samples//2
    alive_count = 0
    dead_count = 0
    counter = 0
    while True:

      # to add the data to the tensors we need an index counter = index
      index = 0

      if alive_count >= target and dead_count >= target:
        break

      # NxN grid
      grid = np.random.choice([0,1],(n,n),p=[1 - rho, rho])

      # we select the top-left corner from which we'll start the MxM subgrid
      i, j = np.random.randint(0, n-m+1, 2)

      # initialize sequence: matrix that will store tau grids
      sequence = np.zeros((tau, m//block_size, m//block_size), dtype=np.float32)

      # loop that iterates the initial grid t_0 times to arrive to the initialization time
      for t in range(t0):

        # we update the grid t0 generations
        grid = next_step_fn(grid)

      # loop that iterates the initial grid tau times according to the game of life rules
      for t in range(tau):

        # choose an MxM subgrid
        subgrid = grid[i:i+m, j:j+m]

        # apply coarse-graining
        coarse_grid = block_average(subgrid, block_size)

        # add to the data
        sequence[t] = coarse_grid

        # update the grid
        grid = next_step_fn(grid)

      # coarse grain the tau+1 grid
      subgrid = grid[i:i+m, j:j+m]
      coarse_grid = block_average(subgrid, block_size)
      label = coarse_grid[m//(2*block_size), m//(2*block_size)]

      if dead_count == target and label == 0:
        for i in range(len(coarse_grid)):
          for j in range(len(coarse_grid)):
            if coarse_grid[i,j] == 1:
              label = coarse_grid[i,j]
              for t in range(tau):
                sequence[t] = np.roll(sequence[t], -i+len(coarse_grid)//2, axis=0)
                sequence[t] = np.roll(sequence[t], -j+len(coarse_grid)//2, axis=1)
              break
          if label == 1:
            break

      elif alive_count == target and label == 1:
        for i in range(len(coarse_grid)):
          for j in range(len(coarse_grid)):
            if coarse_grid[i,j] == 0:
              label = coarse_grid[i,j]
              coarse_grid = np.roll(coarse_grid, -i, axis=0)
              coarse_grid = np.roll(coarse_grid, -j, axis=1)
              break
          if label == 0:
            break

      if int(label) == 1 and alive_count < target:
        self.data.append(sequence)
        self.labels.append(label)
        alive_count += 1

      elif int(label) == 0 and dead_count < target:
        self.data.append(sequence)
        self.labels.append(label)
        dead_count += 1

      counter += 1

    # print(f"Counter : {counter}, Dead count : {dead_count}, Alive count : {alive_count}")
    # data normalization
    if normalize:
      self.normalize()

    # turn into torch tensors
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.labels = torch.tensor(self.labels, dtype=torch.float32)

  def __len__(self):
    """Length of the dataset (how many samples exist)"""
    length = len(self.labels)
    return length

  def __getitem__(self,idx):
    """Retrieve a single sample from the dataset"""

    x = self.data[idx]

    # labels as the y vector
    y = self.labels[idx]

    return x,y

  def normalize(self):
    """Normalize dataset using precomputed mean and std"""

    self.data = np.array(self.data)
    mean_path, std_path = get_stats_paths(self.block_size, self.m, self.n, self.tau, self.rho)
    try:
      mean = np.load(mean_path)
      std = np.load(std_path)
    except:
      raise RuntimeError("Mean and std files not found")

    # avoid NaN
    eps = 1e-6
    std[std < eps] = 1
    self.data = (self.data - mean) / std