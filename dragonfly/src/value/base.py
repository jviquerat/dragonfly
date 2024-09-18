# Generic imports
import numpy as np
import torch

# Custom imports
from dragonfly.src.network.network import net_factory
from dragonfly.src.optimizer.optimizer import opt_factory
from dragonfly.src.loss.loss import loss_factory

###############################################
### Base value
class base_value(torch.nn.Module):
    def __init__(self):
        super(base_value, self).__init__()

    # Get values
    def values(self, obs):
        raise NotImplementedError

    # Network forward pass
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)[0]

    # Get values
    def values(self, x):
        return self.forward(x).reshape(-1, self.out_dim)

    # Save network weights
    def save_weights(self):
        self.weights = self.net.state_dict()

    # Set network weights
    def set_weights(self, weights):
        self.net.load_state_dict(weights)

    # Get current learning rate
    def lr(self):
        return self.opt.get_lr()

    # Reset
    def reset(self):
        self.net.reset()
        self.opt.reset()

        if self.target:
            self.tgt.reset()
            self.copy_tgt()

    # Save
    def save(self, filename):
        torch.save(self.net.state_dict(), filename)

    # Load
    def load(self, filename):
        self.net.load_state_dict(torch.load(filename, weights_only=True))

    # Copy net into tgt
    def copy_tgt(self):
        self.tgt.load_state_dict(self.net.state_dict())
