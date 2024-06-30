# PyTorch imports
import torch
import torch.nn as nn

###############################################
### Alpha SAC loss
class alpha_sac(nn.Module):
    def __init__(self):
        super(alpha_sac, self).__init__()

    # Train
    def train(self, obs, p, log_alpha, tgt_entropy, opt):
        opt.zero_grad()

        # Compute loss
        act, lgp = p.sample(obs)
        loss = log_alpha * (lgp + tgt_entropy)
        loss = -torch.mean(loss)

        # Compute gradients and update
        loss.backward()
        opt.apply_grads()

        return loss.item()

