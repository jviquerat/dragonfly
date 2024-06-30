# Custom imports
from dragonfly.src.network.base import *
import torch
import torch.nn as nn

###############################################
### Autoencoder network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class ae(BaseNetwork):
    def __init__(self, inp_dim, lat_dim, pms):

        # Initialize base class
        super(ae, self).__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.lat_dim = lat_dim

        # Set default values
        self.arch = [64]
        self.actv = "relu"

        # Check inputs
        if hasattr(pms, "arch"):  self.arch = pms.arch
        if hasattr(pms, "actv"):  self.actv = pms.actv

        # Initialize network
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        # Define encoder
        for l in range(len(self.arch)):
            self.enc.append(nn.Linear(inp_dim if l == 0 else self.arch[l-1], self.arch[l]))
            self.enc.append(get_activation(self.actv))
        self.enc.append(nn.Linear(self.arch[-1], self.lat_dim))

        # Define decoder
        for l in range(len(self.arch)):
            self.dec.append(nn.Linear(self.lat_dim if l == 0 else self.arch[-l], self.arch[-l-1]))
            self.dec.append(get_activation(self.actv))
        self.dec.append(nn.Linear(self.arch[0], self.inp_dim))

        # Initialize weights
        dummy = self.forward(torch.ones(1, self.inp_dim))

        # Save initial weights
        self.init_weights = self.state_dict()

    # Network forward pass
    def forward(self, var):
        v = self.encoder(var)
        v = self.decoder(v)
        return v

    # Encoder forward pass
    def encoder(self, var):
        for layer in self.enc:
            var = layer(var)
        return var

    # Decoder forward pass
    def decoder(self, var):
        for layer in self.dec:
            var = layer(var)
        return var

    # Reset weights
    def reset(self):
        self.load_state_dict(self.init_weights)
