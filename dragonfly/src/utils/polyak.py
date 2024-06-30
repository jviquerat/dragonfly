# Generic imports
import torch

###############################################
### Polyak averager for neural networks
class polyak:
    def __init__(self, rho):
        self.rho = rho

    # Update network by polyak average
    def average(self, net, tgt):
        with torch.no_grad():
            for param_v, param_t in zip(net.parameters(), tgt.parameters()):
                param_t.data.copy_(self.rho * param_t.data + (1.0 - self.rho) * param_v.data)

