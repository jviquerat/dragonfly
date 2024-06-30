# PyTorch imports
import torch

###############################################
### Q-policy loss class for DDPG policy update
class q_pol():
    def __init__(self, pms):
        pass

    # Train
    def train(self, obs, p, q, opt):
        # Compute loss
        lgt = obs.shape[0]
        act = p(obs)[0].reshape(lgt, -1)
        cct = torch.cat([obs, act], dim=-1)
        tgt = q(cct)[0].reshape(lgt, -1)
        loss = -torch.mean(tgt)

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss
