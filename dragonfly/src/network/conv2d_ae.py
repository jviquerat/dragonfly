# Generic imports
import math
import torch
import torch.nn as nn

# Custom imports
from dragonfly.src.network.base import *

###############################################
### Autoencoder network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class conv2d_ae(BaseNetwork):
    def __init__(self, inp_dim, lat_dim, pms):

        # Initialize base class
        super(conv2d_ae, self).__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.lat_dim = lat_dim

        # Set default values
        self.trunk         = trunk()
        self.trunk.filters = [64]
        self.trunk.kernels = [3]
        self.trunk.stride  = 1
        self.trunk.actv    = nn.ReLU()
        self.k_init        = nn.init.orthogonal_
        self.original_dim  = None
        self.pooling       = False

        # Check inputs
        if hasattr(pms,       "trunk"):         self.trunk         = pms.trunk
        if hasattr(pms.trunk, "filters"):       self.trunk.filters = pms.trunk.filters
        if hasattr(pms.trunk, "kernels"):       self.trunk.kernels = pms.trunk.kernels
        if hasattr(pms.trunk, "strides"):       self.trunk.stride  = pms.trunk.stride
        if hasattr(pms.trunk, "actv"):          self.trunk.actv    = pms.trunk.actv
        if hasattr(pms,       "k_init"):        self.k_init        = pms.k_init
        if hasattr(pms,       "original_dim"):  self.original_dim  = pms.original_dim
        if hasattr(pms,       "pooling"):       self.pooling       = pms.pooling

        # Specific dimensions
        self.nx    = self.original_dim[0]
        self.ny    = self.original_dim[1]
        self.stack = self.original_dim[2]

        # Initialize network
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.w   = self.nx
        self.h   = self.ny

        # Define encoder
        for l in range(len(self.trunk.filters)):
            if l == 0:
                self.enc.append(nn.Conv2d(in_channels=self.stack,
                                          out_channels=self.trunk.filters[l],
                                          kernel_size=self.trunk.kernels[l],
                                          stride=self.trunk.stride,
                                          padding='same'))
            else:
                self.enc.append(nn.Conv2d(in_channels=self.trunk.filters[l-1],
                                          out_channels=self.trunk.filters[l],
                                          kernel_size=self.trunk.kernels[l],
                                          stride=self.trunk.stride,
                                          padding='same'))
            self.enc.append(get_activation(self.trunk.actv))
            self.w = self.compute_shape(self.w, self.trunk.kernels[l], 2, self.trunk.stride)
            self.h = self.compute_shape(self.h, self.trunk.kernels[l], 2, self.trunk.stride)

            if self.pooling:
                self.enc.append(nn.MaxPool2d(kernel_size=2))
                self.w = self.compute_shape(self.w, 2, 0, 1)
                self.h = self.compute_shape(self.h, 2, 0, 1)

        # Linear layer to bottleneck
        self.enc.append(nn.Flatten())
        self.enc.append(nn.Linear(self.w * self.h * self.trunk.filters[-1], self.lat_dim))

        # Linear layer from bottleneck
        d = self.w * self.h * self.trunk.filters[-1]
        self.dec.append(nn.Linear(self.lat_dim, d))
        self.dec.append(nn.Unflatten(1, (self.trunk.filters[-1], self.w, self.h)))

        # Define decoder
        for l in range(len(self.trunk.filters)-1, -1, -1):
            self.dec.append(nn.ConvTranspose2d(in_channels=self.trunk.filters[l],
                                               out_channels=self.trunk.filters[l-1] if l > 0 else self.stack,
                                               kernel_size=self.trunk.kernels[l],
                                               stride=self.trunk.stride,
                                               padding='same'))
            if l > 0:
                self.dec.append(get_activation(self.trunk.actv))
            else:
                self.dec.append(nn.Sigmoid())

        # Initialize weights
        self.apply(self._init_weights)

        # Save initial weights
        self.init_weights = self.state_dict()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            self.k_init(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        return var.flatten(1)

    # Reset weights
    def reset(self):
        self.load_state_dict(self.init_weights)

    # Compute output shape for conv layers
    # w = width (and height, assuming square images)
    # k = kernel size
    # p = padding
    # s = stride
    # d = dilation
    def compute_shape(self, w, k=3, p=0, s=1, d=1):
        return math.floor((w + 2*p - d*(k - 1) - 1)/s + 1)
