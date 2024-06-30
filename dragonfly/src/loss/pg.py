# PyTorch imports
import torch

###############################################
### PG loss class
class pg():
    def __init__(self, pms):

        # Set default values
        self.ent_coef = 0.01

        # Check inputs
        if hasattr(pms, "ent_coef"): self.ent_coef = pms.ent_coef

    # Train
    def train(self, obs, adv, act, p, opt):
        # Compute loss
        pdf = p.compute_pdf(obs)
        lgp = pdf.log_prob(act)
        loss_pg = torch.multiply(adv, lgp)
        loss_pg = -torch.mean(loss_pg)

        # Compute entropy loss
        entropy = pdf.entropy().reshape(-1)
        entropy = torch.mean(entropy, dim=0)
        loss_entropy = -entropy

        # Compute total loss
        loss = loss_pg + self.ent_coef * loss_entropy

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss