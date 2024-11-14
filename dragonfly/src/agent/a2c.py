# Custom imports
from dragonfly.src.agent.base import *

###############################################
### A2C agent
class a2c(base_agent_on_policy):
    def __init__(self, spaces, n_cpu, size, pms):
        super().__init__(spaces)

        # Initialize from arguments
        self.name   = 'a2c'
        self.n_cpu  = n_cpu
        self.size   = size

        self.p = pol_factory.create(pms.policy.type,
                                    obs_dim   = self.obs_dim(),
                                    obs_shape = self.obs_shape(),
                                    act_dim   = self.act_dim(),
                                    pms       = pms.policy)

        self.v = val_factory.create(pms.value.type,
                                    inp_dim   = self.obs_dim(),
                                    inp_shape = self.obs_shape(),
                                    pms       = pms.value)

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
        act = self.p.reshape_actions(act)
        adv = tf.reshape(adv, [-1])
        self.p.loss.train(obs, adv, act, self.p, self.p.opt)

        # Train v network
        tgt = tf.reshape(tgt, [-1])
        self.v.loss.train(obs, tgt, self.v.net, self.v.opt)
