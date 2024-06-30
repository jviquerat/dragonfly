# PyTorch imports
import torch

###############################################
### MSE loss class for DQN-style q networks
class mse_dqn():
    def __init__(self, pms):
        pass

    # Train
    def train(self,
              obs, nxt, act, rwd, trm,
              gamma, q, q_tgt, opt):
        # Compute target
        tgt = q_tgt(nxt)[0].reshape(rwd.size(0), -1)
        tgt = torch.max(tgt, dim=1)[0]
        tgt = tgt.reshape(-1, 1)
        trm = torch.clamp(trm, 0.0, 1.0)
        tgt = rwd + trm * gamma * tgt

        # Compute loss
        val = q(obs)[0].reshape(tgt.size(0), -1)
        val = torch.gather(val, 1, act.long())
        diff = torch.square(tgt - val)
        loss = torch.mean(diff)

        # Apply gradients
        opt.zero_grad()
        loss.backward()
        opt.apply_grads()

        return loss
