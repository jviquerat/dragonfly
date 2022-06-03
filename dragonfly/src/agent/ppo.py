# Custom imports
from dragonfly.src.agent.base            import *
from dragonfly.src.terminator.terminator import *
from dragonfly.src.utils.buff            import *
from dragonfly.src.utils.counter         import *

###############################################
### PPO agent
class ppo(base_agent):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name      = 'ppo'
        self.act_dim   = act_dim
        self.obs_dim   = obs_dim
        self.n_cpu     = n_cpu
        self.size      = size

        # Build policies
        if (pms.policy.loss.type != "surrogate"):
            warning("ppo", "__init__",
                    "Loss type for ppo agent is not surrogate")

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

        # Create buffers
        self.buff = buff(self.n_cpu,
                         ["obs", "nxt", "act", "lgp",
                          "rwd", "dne", "stp", "trm", "bts"],
                         [obs_dim, obs_dim, self.pol_dim, 1, 1, 1, 1, 1, 1])
        self.gbuff = gbuff(self.size,
                           ["obs", "act", "adv", "tgt", "lgp"],
                           [obs_dim, self.pol_dim, 1, 1, 1])

        # Initialize counter
        self.counter = counter(self.n_cpu)

        # Initialize terminator
        self.terminator = terminator_factory.create(pms.terminator.type,
                                                    n_cpu = self.n_cpu,
                                                    pms   = pms.terminator)

    # Get actions
    def actions(self, obs):

        # Get actions and associated log-prob
        act, lgp = self.p_net.actions(obs)

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ppo", "get_actions",
                  "Detected NaN in generated actions")

        # Store log-prob
        self.buff.store(["lgp"], [ lgp ])

        return act

    # Control (deterministic actions)
    def control(self, obs):

        return self.p_net.control(obs)

    # Finalize buffers before training
    def returns(self, obs, nxt, act, rwd, trm, bts):

        # Get current and next values
        cval = self.v_net.values(obs)
        nval = self.v_net.values(nxt)

        # Compute advantages
        tgt, adv = self.retrn.compute(rwd, cval, nval, trm, bts)

        return tgt, adv

    # Prepare training data
    def prepare_data(self, size):

        names = ["obs", "adv", "tgt", "act", "lgp"]
        self.data = self.gbuff.get_buffers(names, size)
        lgt = len(self.data["obs"])

        return lgt

    # Training
    def train(self, start, end):

        obs = self.data["obs"][start:end]
        act = self.data["act"][start:end]
        adv = self.data["adv"][start:end]
        tgt = self.data["tgt"][start:end]
        lgp = self.data["lgp"][start:end]

        # Train policy
        act = self.p_net.reshape_actions(act)
        adv = tf.reshape(adv, [-1])
        lgp = tf.reshape(lgp, [-1])
        self.p_net.loss.train(obs, adv, act, lgp, self.p_net)

        # Train v network
        tgt = tf.reshape(tgt, [-1])
        self.v_net.loss.train(obs, tgt, self.v_net)

    # Agent reset
    def reset(self):

        self.counter.reset()
        self.p_net.reset()
        self.v_net.reset()

    # Store transition
    def store(self, obs, act, rwd, nxt, dne):

        stp = self.counter.ep_step
        self.buff.store(["obs", "act", "rwd", "nxt", "dne", "stp"],
                        [ obs,   act,   rwd,   nxt,   dne,   stp ])
        self.counter.update(rwd)

    # Actions to execute before the inner training loop
    def pre_loop(self):

        self.buff.reset()

    # Actions to execute after the inner training loop
    def post_loop(self):

        self.terminator.terminate(self.buff)
        names = ["obs", "nxt", "act", "lgp", "rwd", "trm", "bts"]
        data  = self.buff.serialize(names)
        gobs, gnxt, gact, glgp, grwd, gtrm, gbts = (data[name] for name in names)
        gtgt, gadv = self.returns(gobs, gnxt, gact, grwd, gtrm, gbts)

        self.gbuff.store(["obs", "adv", "tgt", "act", "lgp"],
                         [gobs,  gadv,  gtgt,  gact,  glgp ])

    # Save agent parameters
    def save(self, filename):

        self.p_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.p_net.load(filename)
