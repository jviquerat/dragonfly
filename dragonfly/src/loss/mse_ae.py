# PyTorch imports
import torch

###############################################
### MSE loss class for auto-encoder
class mse_ae():
    def __init__(self, pms):
        pass

    # Train
    def train(self, x, ae):
        # Compute loss
        y = ae(x)
        diff = torch.square(y - x)
        loss = torch.mean(diff)

        # Apply gradients
        ae.opt.zero_grad()
        loss.backward()
        ae.opt.apply_grads()

        return loss
