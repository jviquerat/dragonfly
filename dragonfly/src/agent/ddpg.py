# Custom imports
from dragonfly.src.agent.base import *

###############################################
### DDPG agent
class ddpg(base_agent):
    def __init__(self, obs_dim, act_dim, n_cpu, pms):

        # Initialize from arguments
        self.name      = 'ppo'
        self.act_dim   = act_dim
        self.obs_dim   = obs_dim
        self.n_cpu     = n_cpu
        self.rho       = pms.rho

        # Build policies
        if (pms.policy.type != "deterministic"):
            error("ppo", "__init__",
                  "Policy type for ddpg agent is not deterministic")

        self.p_net = pol_factory.create(pms.policy.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.policy)
        self.p_tgt = pol_factory.create(pms.policy.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.policy)
        self.p_tgt.net.set_weights(self.p_net.net.get_weights())

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.pol_dim = self.policy.store_dim

        # Build values
        if (pms.value.type != "q_value"):
            error("ddpg", "__init__",
                  "Value type for ddpg agent is not q_value")

        self.q_net = val_factory.create(pms.value.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.value)
        self.q_tgt = val_factory.create(pms.value.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.value)
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())

        # polyak averager
        self.polyak = polyak(self.rho)

        # Build advantage
        #self.retrn = retrn_factory.create(pms.retrn.type,
        #                                  pms = pms.retrn)

    # Get actions
    def get_actions(self, obs):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, self.pol_dim],
                       dtype=self.policy.store_type)
        lgp = np.zeros([self.n_cpu, 1])

        # Loop over cpus
        for i in range(self.n_cpu):
            act[i,:] = self.policy.get_actions(obs[i])

        # Reshape actions depending on policy type
        act = np.reshape(act, (-1,self.pol_dim))

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ppo", "get_actions", "Detected NaN in generated actions")

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, self.pol_dim],
                       dtype=self.policy.store_type)

        # Loop over cpus
        for i in range(self.n_cpu):
            act[i,:] = self.policy.control(obs[i])

        # Reshape actions depending on policy type
        if (self.policy.kind == "discrete"):
            act = np.reshape(act, (-1))
        if (self.policy.kind == "continuous"):
            act = np.reshape(act, (-1,self.pol_dim))

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

    # Compute entropy
    def entropy(self, obs):

        return self.policy.entropy(obs)

    # Training
    def train(self, btc_obs, btc_act, btc_adv, btc_tgt, btc_lgp, size):

        self.policy.train(btc_obs, btc_adv, btc_act, btc_lgp)
        self.q_net.train(btc_obs, btc_tgt, size)



    # Reset
    def reset(self):

        self.p_net.reset()
        self.q_net.reset()
        self.p_tgt.net.set_weights(self.p_net.net.get_weights())
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())

    # Save agent parameters
    def save(self, filename):

        self.policy.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.policy.load(filename)
