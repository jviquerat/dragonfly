# Custom imports
from dragonfly.src.agent.base import *
from dragonfly.src.srl.srl    import *

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
        self.init_srl(pms, obs_dim, 100*size)
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

    # Pre-process observations using srl
    def process_obs(self, obs):

        return self.srl.process(obs)

    # Initialize srl
    def init_srl(self, pms, obs_dim, buff_size):

        # Check inputs
        if not hasattr(pms, "srl"):
            pms.srl  = None
            srl_type = "dummy"
        else: srl_type = pms.srl.type

        # Create srl
        self.srl = srl_factory.create(srl_type,
                                      obs_dim   = obs_dim,
                                      buff_size = buff_size,
                                      pms       = pms.srl)

    # Get actions
    def actions(self, obs):

        pobs = self.process_obs(obs)

        # Get actions and associated log-prob
        self.timer_actions.tic()
        act, lgp = self.p.actions(pobs)

        # Check for NaNs
        if (np.isnan(act).any()):
            error("a2c", "get_actions",
                  "Detected NaN in generated actions")

        # Store log-prob
        self.buff.store(["lgp"], [lgp])

        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        pobs = self.process_obs(obs)

        return self.p.control(pobs)

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

    # Actions to execute after the inner training loop
    def post_loop(self, style=None):

        # For buffer-style training, the last step of each buffer
        # must be bootstraped to mimic a continuing episode
        if ((style == "buffer") and (self.term.type == "bootstrap")):
            for i in range(self.n_cpu):
                done = (self.buff.data["trm"].buff[i][-1] == 0.0)
                if (not done):
                    self.buff.data["trm"].buff[i][-1] = 2.0

        names = ["obs", "nxt", "act", "lgp", "rwd", "trm"]
        data  = self.buff.serialize(names)
        gobs, gnxt, gact, glgp, grwd, gtrm = (data[name] for name in names)

        pgobs = self.process_obs(gobs)
        pgnxt = self.process_obs(gnxt)

        gtgt, gadv = self.returns(pgobs, pgnxt, grwd, gtrm)

        self.gbuff.store(self.gnames, [gobs, gact, gadv, gtgt, glgp])

    # Agent reset
    def reset(self):

        self.p.reset()
        self.v.reset()
        self.buff.reset()
        self.gbuff.reset()
        self.srl.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(["obs", "nxt", "act", "rwd", "trm"],
                        [ obs,   nxt,   act,   rwd,   trm ])

        # Store in SRL buffer
        self.srl.store_obs(obs)

    # Save value parameters
    def save_value(self, filename):

        self.v.save(filename)

    # Load value parameters
    def load_value(self, filename):

        self.v.load(filename)
