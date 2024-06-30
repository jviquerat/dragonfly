# PyTorch imports
import torch

###############################################
### MSE loss class for policy gradient-style value networks
class mse_pg():
    def __init__(self, pms):
        pass

    # Train
    def train(self, obs, tgt, net, opt):
        # Compute loss
        val = net(obs)[0].reshape(tgt.size(0))
        diff = torch.square(tgt - val)
        loss = torch.mean(diff)

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss
