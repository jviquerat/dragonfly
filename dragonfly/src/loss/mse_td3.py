# PyTorch imports
import torch

###############################################
### MSE loss class for TD3-style q networks
class mse_td3():
    def __init__(self, pms):
        pass

    # Train
    def train(self,
              obs, nxt, act, rwd, trm, gamma, sigma, clp,
              p_tgt, q, q1_tgt, q2_tgt, opt):
        # Compute target
        nac = p_tgt(nxt)[0].reshape(rwd.size(0), -1)
        nse = torch.normal(0.0, sigma, size=nac.shape)
        nse = torch.clamp(nse, -clp, clp)
        nac = torch.clamp(nac + nse, -1.0, 1.0)
        nct = torch.cat([nxt, nac], dim=-1)
        tgt1 = q1_tgt(nct)[0].reshape(-1, 1)
        tgt2 = q2_tgt(nct)[0].reshape(-1, 1)
        tgt = torch.min(tgt1, tgt2)
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
