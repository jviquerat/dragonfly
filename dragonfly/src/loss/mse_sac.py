# PyTorch imports
import torch

###############################################
### MSE loss class for SAC-style q networks
class mse_sac():
    def __init__(self, pms):
        pass

    # Train
    def train(self,
              obs, nxt, act, rwd, trm, gamma, alpha,
              p, q, q1_tgt, q2_tgt, opt):
        # Compute target
        nac, lgp = p.sample(nxt)
        nct = torch.cat([nxt, nac], dim=-1)
        tgt1 = q1_tgt(nct)[0].reshape(-1, 1)
        tgt2 = q2_tgt(nct)[0].reshape(-1, 1)
        tgt = torch.min(tgt1, tgt2)
        trm = torch.clamp(trm, 0.0, 1.0)
        tgt = rwd + trm * gamma * (tgt - alpha * lgp)

        # Compute loss
        oac = torch.cat([obs, act], dim=-1)
        val = q(oac)[0].reshape(-1, 1)
        diff = torch.square(tgt - val)
        loss = torch.mean(diff)

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss
