# PyTorch imports
import torch

###############################################
### Q-policy loss class for SAC policy update
class q_pol_sac():
    def __init__(self, pms):
        pass

    # Train
    def train(self, obs, p, q1, q2, alpha, opt):
        # Compute loss
        act, lgp = p.sample(obs)
        cct = torch.cat([obs, act], dim=-1)
        tgt1 = q1(cct)[0].reshape(-1, 1)
        tgt2 = q2(cct)[0].reshape(-1, 1)
        tgt = torch.min(tgt1, tgt2)
        loss = -torch.mean(tgt - alpha * lgp)

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss
