# Custom imports
from dragonfly.src.agent.base import *

###############################################
### PPO-SRL agent
class ppo_srl(base_agent_on_policy):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name      = 'ppo'
        self.act_dim   = act_dim
        self.obs_dim  = obs_dim
        self.n_cpu     = n_cpu
        self.size      = size

        # Initialize srl class
        self.init_srl(pms, obs_dim, 1000*size)
        self.latent_dim = self.srl.latent_dim

        # Build policies
        if (pms.policy.loss.type != "surrogate"):
            warning("ppo", "__init__",
                    "Loss type for ppo agent is not surrogate")

        self.p = pol_factory.create(pms.policy.type,
                                    obs_dim = self.latent_dim,
                                    act_dim = act_dim,
                                    pms     = pms.policy)

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.pol_dim = self.p.store_dim

        # Build values
        if (pms.value.type != "v_value"):
            warning("ppo", "__init__",
                    "Value type for ppo agent is not v_value")

        self.v = val_factory.create(pms.value.type,
                                    inp_dim = self.latent_dim,
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
        lgp = self.data["lgp"][start:end]

        # Train policy
        pobs = self.process_obs(obs)
        act = self.p.reshape_actions(act)
        adv = tf.reshape(adv, [-1])
        lgp = tf.reshape(lgp, [-1])
        self.p.loss.train(pobs, adv, act, lgp, self.p, self.p.opt)

        # Train v network
        tgt = tf.reshape(tgt, [-1])
        self.v.loss.train(pobs, tgt, self.v.net, self.v.opt)

    # Save value parameters
    def save_value(self, filename):

        self.v.save(filename)

    # Load value parameters
    def load_value(self, filename):

        self.v.load(filename)
