# PyTorch imports
import torch

# Custom imports
from dragonfly.src.core.constants import *

###############################################
### Surrogate loss class
class surrogate():
    def __init__(self, pms):

        # Set default values
        self.pol_clip = 0.2
        self.ent_coef = 0.01

        # Check inputs
        if hasattr(pms, "pol_clip"): self.pol_clip = pms.pol_clip
        if hasattr(pms, "ent_coef"): self.ent_coef = pms.ent_coef

    # Train
    def train(self, obs, adv, act, plg, p, opt):
        # Compute ratio of probabilities
        pdf = p.compute_pdf(obs)
        lgp = pdf.log_prob(act)
        ratio = torch.exp(lgp - plg)

        # Compute actor loss
        p1 = torch.multiply(adv, ratio)
        p2 = torch.clamp(ratio, 1.0 - self.pol_clip, 1.0 + self.pol_clip)
        p2 = torch.multiply(adv, p2)
        loss_surrogate = -torch.mean(torch.min(p1, p2))

        # Compute entropy loss
        entropy = pdf.entropy().reshape(-1)
        entropy = torch.mean(entropy, dim=0)
        loss_entropy = -entropy

        # Compute total loss
        loss = loss_surrogate + self.ent_coef * loss_entropy

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss
