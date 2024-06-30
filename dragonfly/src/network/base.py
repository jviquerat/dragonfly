# Generic imports
import os
import warnings

# Filter warning messages
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings('ignore', category=FutureWarning)

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

# Custom imports
from dragonfly.src.utils.error import error, warning
from dragonfly.src.network.tree import trunk, heads

###############################################
### Base network
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    # Network forward pass
    def forward(self, x):
        raise NotImplementedError

    # Reset weights
    def reset(self):
        raise NotImplementedError

    # Return trainable parameters
    def trainables(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def create_dense_layer(self, in_features, out_features, init_func, activation):
        layer = nn.Linear(in_features, out_features)
        init_func(layer.weight)
        nn.init.zeros_(layer.bias)
        return nn.Sequential(
            layer,
            get_activation(activation)
        )

def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "swish" or activation_name == "silu":
        return nn.SiLU()  # SiLU is PyTorch's implementation of Swish
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "linear" or activation_name is None:
        return nn.Identity()
    elif activation_name == "softmax":
            return nn.Softmax()
    elif activation_name == "sigmoid":
            return nn.Sigmoid()
    elif activation_name == "selu":
            return nn.SELU()
    elif activation_name == "gelu":
            return nn.GELU()
    # Add more activations as needed
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")