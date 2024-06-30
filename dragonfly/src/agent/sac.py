# Custom imports
from dragonfly.src.agent.base import *
from dragonfly.src.utils.polyak import polyak
import torch
import numpy as np

###############################################
### SAC agent
class sac(base_agent_off_policy):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name       = 'sac'
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.n_cpu      = n_cpu
        self.mem_size   = size
        self.gamma      = pms.gamma
        self.alpha      = pms.alpha
        self.rho        = pms.rho
        self.n_warmup   = pms.n_warmup
        self.n_filling  = pms.n_filling

        # Local variables
        self.step = 0

        # Build policies
        if (pms.policy.loss.type != "q_pol_sac"):
            error("sac", "__init__",
                  "Policy loss type for sac agent is not q_pol_sac")
        if (pms.policy.type != "tanh_normal"):
            error("sac", "__init__",
                  "Policy type for sac agent is not tanh_normal")

        self.p = pol_factory.create(pms.policy.type,
                                    obs_dim = obs_dim,
                                    act_dim = act_dim,
                                    pms     = pms.policy)

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.pol_dim = self.p.store_dim

        # Build values
        if (pms.value.type != "q_value"):
            error("sac", "__init__",
                  "Value type for sac agent is not q_value")

        if (pms.value.loss.type != "mse_sac"):
            error("td3", "__init__",
                  "Loss type for sac agent is not mse_sac")

        self.q1 = val_factory.create(pms.value.type,
                                     inp_dim = obs_dim+act_dim,
                                     out_dim = 1,
                                     pms     = pms.value,
                                     target  = True)
        self.q2 = val_factory.create(pms.value.type,
                                     inp_dim = obs_dim+act_dim,
                                     out_dim = 1,
                                     pms     = pms.value,
                                     target  = True)

        # Polyak averager
        self.polyak = polyak(self.rho)

        # Create buffers
        self.create_buffers(act_dim=self.pol_dim)

        # Initialize termination
        self.term = termination_factory.create(pms.termination.type,
                                               n_cpu = self.n_cpu,
                                               pms   = pms.termination)

        # Initialize timer
        self.timer_actions = timer("actions  ")

    # Get actions
    def actions(self, obs):

        self.timer_actions.tic()
        if (self.step < self.n_warmup):
            act = np.random.uniform(-1.0, 1.0, (self.n_cpu, self.act_dim))
            act = act.astype(np.float32)
        else:
            act, _ = self.p.actions(torch.tensor(obs, dtype=torch.float32))
            #act = act.detach().numpy()

        self.step += 1

        # Check for NaNs
        if np.isnan(act).any():
            error("sac", "get_actions",
                  "Detected NaN in generated actions")
        self.timer_actions.toc()

        return act

    # Training
    def train(self, start, end):

        obs = self.data["obs"][start:end]
        nxt = self.data["nxt"][start:end]
        act = self.data["act"][start:end]
        rwd = self.data["rwd"][start:end]
        trm = self.data["trm"][start:end]

        size = end - start
        act = self.p.reshape_actions(act)
        rwd = rwd.reshape(size, -1)
        trm = trm.reshape(size, -1)

        # Train q network
        self.q1.loss.train(obs, nxt, act, rwd, trm, self.gamma,
                           self.alpha, self.p, self.q1.net,
                           self.q1.tgt, self.q2.tgt, self.q1.opt)
        self.q2.loss.train(obs, nxt, act, rwd, trm, self.gamma,
                           self.alpha, self.p, self.q2.net,
                           self.q1.tgt, self.q2.tgt, self.q2.opt)

        # Train policy network
        self.p.loss.train(obs, self.p, self.q1.net, self.q2.net,
                          self.alpha, self.p.opt)

        # Update target networks
        self.polyak.average(self.q1.net, self.q1.tgt)
        self.polyak.average(self.q2.net, self.q2.tgt)

    # Reset
    def reset(self):

        self.step = 0
        self.p.reset()
        self.q1.reset()
        self.q2.reset()
        self.buff.reset()
        self.gbuff.reset()
