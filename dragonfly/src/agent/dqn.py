# Custom imports
from dragonfly.src.agent.base import *
import torch
import random
import numpy as np

###############################################
### DQN agent
class dqn(base_agent_off_policy):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name       = 'dqn'
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.n_cpu      = n_cpu
        self.mem_size   = size
        self.gamma      = pms.gamma
        self.tgt_update = pms.tgt_update

        # Local variables
        self.tgt_step   = 0

        # Initialize random limit
        self.eps = decay_factory.create(pms.exploration.type,
                                        pms = pms.exploration)

        # Build values
        if (pms.value.type != "q_value"):
            error("dqn", "__init__",
                  "Value type for dqn agent is not q_value")

        if (pms.value.loss.type != "mse_dqn"):
            error("dqn", "__init__",
                  "Loss type for dqn agent is not mse_dqn")

        self.q = val_factory.create(pms.value.type,
                                    inp_dim = obs_dim,
                                    out_dim = act_dim,
                                    pms     = pms.value,
                                    target  = True)

        # Create buffers
        self.create_buffers(act_dim=1)

        # Initialize termination
        self.term = termination_factory.create(pms.termination.type,
                                               n_cpu = self.n_cpu,
                                               pms   = pms.termination)

        # Initialize timer
        self.timer_actions = timer("actions  ")

    # Get actions
    def actions(self, obs):

        self.timer_actions.tic()
        act = torch.zeros(self.n_cpu, 1, dtype=torch.long)

        for i in range(self.n_cpu):
            self.eps.decay()
            p = random.uniform(0, 1)
            if (p < self.eps.get()):
                act[i] = random.randrange(0, self.act_dim)
            else:
                cob = torch.tensor(np.array([obs[i]]), dtype=torch.float32)
                val = self.q.values(cob)
                act[i] = torch.argmax(val)

        act = act.reshape(-1).numpy()
        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        val = self.q.values(torch.tensor(obs, dtype=torch.float32))
        act = torch.argmax(val, dim=1).reshape(-1).numpy()

        return act

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < size): return lgt, False

        self.data = self.gbuff.get_batches(self.names, size)
        return lgt, True

    # Training
    def train(self, start, end):

        obs = self.data["obs"][start:end]
        nxt = self.data["nxt"][start:end]
        act = self.data["act"][start:end]
        rwd = self.data["rwd"][start:end]
        trm = self.data["trm"][start:end]

        size = end - start
        act = act.reshape(size, -1)
        rwd = rwd.reshape(size, -1)
        trm = trm.reshape(size, -1)
        act = act.long()

        self.q.loss.train(obs, nxt, act, rwd, trm,
                          self.gamma, self.q.net,
                          self.q.net, self.q.opt)

        # Update target networks
        if (self.tgt_step == self.tgt_update):
            self.q.copy_tgt()
            self.tgt_step  = 0
        else:
            self.tgt_step += 1

    # Reset
    def reset(self):

        self.tgt_step = 0
        self.q.reset()
        self.buff.reset()
        self.gbuff.reset()
        self.eps.reset()

    # Save parameters
    # This function overrides the base_agent::save_policy
    def save_policy(self, filename):

        self.q.save(filename)

    # Load parameters
    # This function overrides the base_agent::load_policy
    def load_policy(self, folder):

        filename = folder + '/' + self.name
        self.q.load(filename)
