# Generic imports
import os
import warnings

# Filter warning messages
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

# Custom imports
from dragonfly.src.utils.error import error, warning
from dragonfly.src.network.tree import trunk, heads


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


def create_dense_layer(in_features, out_features, init_func, activation):
    layer = nn.Linear(in_features, out_features)
    init_func(layer.weight)
    nn.init.zeros_(layer.bias)
    return nn.Sequential(layer, get_activation(activation))


def build_trunk(
    in_size: int, hidden_sizes: list[int], init_func, activation_function: str
):
    new_hidden_sizes = [in_size] + hidden_sizes
    layers = [
        create_dense_layer(
            new_hidden_sizes[k], new_hidden_sizes[k + 1], init_func, activation_function
        )
        for k in range(0, len(new_hidden_sizes) - 1)
    ]
    return nn.Sequential(*layers)


def build_heads(
    nb_heads: int,
    in_size: int,
    hidden_sizes: list[list[int]],
    out_sizes: list[int],
    activation_functions: list[str],
    finals: list[str],
    k_init,
    k_init_final,
):
    heads = []
    for h in range(nb_heads):
        new_hidden_sizes = [in_size] + hidden_sizes[h]
        layers = [
            create_dense_layer(
                new_hidden_sizes[k],
                new_hidden_sizes[k + 1],
                k_init,
                activation_functions[h],
            )
            for k in range(0, len(new_hidden_sizes) - 1)
        ]
        layers.append(
            create_dense_layer(
                new_hidden_sizes[-1], out_sizes[h], k_init_final, finals[h]
            )
        )

        heads.append(nn.Sequential(*layers))

    return heads


###############################################
### Base network
class BaseNetwork(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(BaseNetwork, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk = trunk()
        self.heads = heads()

        self.k_init = lambda x: orthogonal_(x, gain=1.0)
        self.k_init_final = lambda x: orthogonal_(x, gain=0.0)

    # Network forward pass
    def forward(self, x):
        raise NotImplementedError

    # Reset weights
    def reset(self):
        raise NotImplementedError

    # Return trainable parameters
    def trainables(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _fetch_input(self, pms, conv: bool = False):
        if hasattr(pms, "trunk"):
            self.trunk = pms.trunk
        if hasattr(pms.trunk, "arch"):
            self.trunk.arch = pms.trunk.arch
        if hasattr(pms.trunk, "actv"):
            self.trunk.actv = pms.trunk.actv
        if hasattr(pms, "heads"):
            self.heads = pms.heads
        if hasattr(pms.heads, "nb"):
            self.heads.nb = pms.heads.nb
        if hasattr(pms.heads, "arch"):
            self.heads.arch = pms.heads.arch
        if hasattr(pms.heads, "actv"):
            self.heads.actv = pms.heads.actv
        if hasattr(pms.heads, "final"):
            self.heads.final = pms.heads.final
        if hasattr(pms, "k_init"):
            self.k_init = pms.k_init
        if hasattr(pms, "k_init_final"):
            self.k_init_final = pms.k_init_final

        if conv:
            if hasattr(pms.trunk, "kernels"):
                self.trunk.kernels = pms.trunk.kernels
            if hasattr(pms.trunk, "strides"):
                self.trunk.stride = pms.trunk.strides
            if hasattr(pms, "pooling"):
                self.pooling = pms.pooling
            if hasattr(pms.trunk, "k_size"):
                self.trunk.k_size = pms.trunk.k_size
            if hasattr(pms, "original_dim"):
                self.original_dim = pms.original_dim

    def _build_trunk(self):
        return build_trunk(
            in_size=self.inp_dim,
            hidden_sizes=self.trunk.arch,
            init_func=self.k_init,
            activation_function=self.trunk.actv,
        )

    def _build_heads(self):
        heads = build_heads(
            nb_heads=self.heads.nb,
            in_size=self.trunk.arch[-1],
            hidden_sizes=self.heads.arch,
            out_sizes=self.out_dim,
            activation_functions=self.heads.actv,
            finals=self.heads.final,
            k_init=self.k_init,
            k_init_final=self.k_init_final,
        )
        for h, head in enumerate(heads):
            setattr(self, f"head_{h}", head)
