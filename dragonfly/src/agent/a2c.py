# Custom imports
from dragonfly.src.agent.base import *

###############################################
### A2C agent
class a2c(base_agent_on_policy):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name      = 'a2c'
        self.act_dim   = act_dim
        self.obs_dim   = obs_dim
        self.n_cpu     = n_cpu
        self.size      = size

        # Build policies
        if (pms.policy.loss.type != "pg"):
            warning("a2c", "__init__",
                    "Loss type for a2c agent is not pg")

        self.p_net = pol_factory.create(pms.policy.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.policy)

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.pol_dim = self.p_net.store_dim

        # Build values
        if (pms.value.type != "v_value"):
            warning("ppo", "__init__",
                    "Value type for ppo agent is not v_value")

        self.v_net = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim,
                                        pms     = pms.value)

        # Build advantage
        self.retrn = retrn_factory.create(pms.retrn.type,
                                          pms = pms.retrn)

        # Create storage buffers
        self.create_buffers()

        # Initialize terminator
        self.term = termination_factory.create(pms.termination.type,
                                               n_cpu = self.n_cpu,
                                               pms   = pms.termination)

        # Initialize timer
        self.timer_actions = timer("actions  ")

    # Training
    def train(self, start, end):

        obs = self.data["obs"][start:end]
        act = self.data["act"][start:end]
        adv = self.data["adv"][start:end]
        tgt = self.data["tgt"][start:end]

        # Train policy
        act = self.p_net.reshape_actions(act)
        adv = tf.reshape(adv, [-1])
        self.p_net.loss.train(obs, adv, act, self.p_net)

        # Train v network
        tgt = tf.reshape(tgt, [-1])
        self.v_net.loss.train(obs, tgt, self.v_net)
