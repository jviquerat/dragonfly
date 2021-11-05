# Generic imports
import math
import copy
import numpy as np

# Custom imports
from dragonfly.src.policy.policy  import *
from dragonfly.src.value.value    import *
from dragonfly.src.retrn.retrn    import *
from dragonfly.src.core.constants import *
from dragonfly.src.utils.error    import *

###############################################
### PPO agent
class ppo():
    def __init__(self, obs_dim, act_dim, n_cpu, pms):

        # Initialize from arguments
        self.name      = 'ppo'
        self.act_dim   = act_dim
        self.obs_dim   = obs_dim
        self.n_cpu     = n_cpu

        # Build policies
        pms.policy.save = True
        if (pms.policy.loss.type != "surrogate"):
            warning("ppo", "__init__",
                    "Loss type for ppo agent is not surrogate")

        self.policy = pol_factory.create(pms.policy.type,
                                         obs_dim = obs_dim,
                                         act_dim = act_dim,
                                         pms     = pms.policy)

        # pol_act_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.pol_act_dim = self.policy.store_dim

        # Build values
        if (pms.value.type != "v_value"):
            warning("ppo", "__init__",
                    "Value type for ppo agent is not v_value")
        self.v_value = val_factory.create(pms.value.type,
                                          obs_dim = obs_dim,
                                          pms     = pms.value)

        # Build advantage
        self.retrn = retrn_factory.create(pms.retrn.type,
                                          pms = pms.retrn)

    # Get actions
    def get_actions(self, observations):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, self.pol_act_dim],
                       dtype=self.policy.store_type)

        # Loop over cpus
        for i in range(self.n_cpu):
            obs      = observations[i]
            act[i,:] = self.policy.get_actions(obs)

        # Reshape actions depending on policy type
        if (self.policy.kind == "discrete"):
            act = np.reshape(act, (-1))
        if (self.policy.kind == "continuous"):
            act = np.reshape(act, (-1,self.pol_act_dim))

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ppo", "get_actions", "Detected NaN in generated actions")

        return act

    # Finalize buffers before training
    def compute_returns(self, obs, nxt, act, rwd, trm, bts):

        # Get current and next values
        crt_val = self.v_value.get_values(obs)
        nxt_val = self.v_value.get_values(nxt)

        # Compute advantages
        tgt, adv = self.retrn.compute(rwd, crt_val, nxt_val, trm, bts)

        return tgt, adv

    # Training
    def train(self, btc_obs, btc_act, btc_adv, btc_tgt, size):

        self.policy.train(btc_obs, btc_adv, btc_act)
        self.v_value.train(btc_obs, btc_tgt, size)

    # Reset
    def reset(self):

        self.policy.reset()
        self.v_value.reset()
