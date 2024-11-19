# Custom imports
from dragonfly.src.agent.base import *

###############################################
### PPO agent
class ppo(base_agent_on_policy):
    def __init__(self, spaces, n_cpu, mem_size, n_stp_unroll, pms):
        super().__init__(spaces)

        # Initialize from arguments
        self.name         = 'ppo'
        self.n_cpu        = n_cpu
        self.mem_size     = mem_size
        self.n_stp_unroll = n_stp_unroll

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
        lgp = self.data["lgp"][start:end]

        # Train policy
        act = self.p.reshape_actions(act)
        adv = tf.reshape(adv, [-1])
        lgp = tf.reshape(lgp, [-1])
        self.p.loss.train(obs, adv, act, lgp, self.p, self.p.opt)

        # Train v network
        tgt = tf.reshape(tgt, [-1])
        self.v.loss.train(obs, tgt, self.v.net, self.v.opt)

    # Save value parameters
    def save_value(self, filename):

        self.v.save(filename)

    # Load value parameters
    def load_value(self, filename):

        self.v.load(filename)
