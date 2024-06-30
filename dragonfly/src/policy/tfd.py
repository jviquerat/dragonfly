# Generic imports
import os
import math
import numpy as np
import warnings

# Import PyTorch and filter warning messages
import torch
import torch.distributions as dist

# Set warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress CUDA warnings if not using GPU
if not torch.cuda.is_available():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set default tensor type to float32
torch.set_default_tensor_type(torch.FloatTensor)

# Define alias for distributions (similar to tfp.distributions)
tfd = dist

# Set PyTorch to behave deterministically
torch.use_deterministic_algorithms(True)
