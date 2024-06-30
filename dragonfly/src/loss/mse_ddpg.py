# PyTorch imports
import torch

###############################################
### MSE loss class for DDPG-style q networks
class mse_ddpg():
    def __init__(self, pms):
        pass

    # Train
    def train(self,
              obs, nxt, act, rwd, trm, gamma,
              p_tgt, q, q_tgt, opt):
        # Compute target
        nac = p_tgt(nxt)[0].reshape(rwd.size(0), -1)
        nct = torch.cat([nxt, nac], dim=-1)
        tgt = q_tgt(nct)[0].reshape(-1, 1)
        trm = torch.clamp(trm, 0.0, 1.0)
        tgt = rwd + trm * gamma * tgt

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