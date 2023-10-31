# Custom imports
from dragonfly.src.agent.base import *

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

        # Initialize base class
        self.init_srl(pms, self.obs_dim, 1000*size)
        obs_dim = self.obs_dim

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
        self.lnames = ["obs", "nxt", "act", "lgp", "rwd", "trm"]
        self.lsizes = [obs_dim, obs_dim, self.pol_dim, 1, 1, 1]
        self.buff   = buff(self.n_cpu, self.lnames, self.lsizes)

        self.gnames = ["obs", "act", "adv", "tgt", "lgp"]
        self.gsizes = [obs_dim, self.pol_dim, 1, 1, 1]
        self.gbuff  = gbuff(self.size, self.gnames, self.gsizes)

        # Initialize terminator
        self.term = termination_factory.create(pms.termination.type,
                                               n_cpu = self.n_cpu,
                                               pms   = pms.termination)

        # Initialize timer
        self.timer_actions = timer("actions  ")

    # Get actions
    def actions(self, obs):

        pobs = super().process_obs(obs)

        # Get actions and associated log-prob
        self.timer_actions.tic()
        act, lgp = self.p_net.actions(pobs)

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ppo", "get_actions",
                  "Detected NaN in generated actions")

        # Store log-prob
        self.buff.store(["lgp"], [lgp])
        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        pobs = super().process_obs(obs)

        return self.p_net.control(pobs)

    # Finalize buffers before training
    def returns(self, obs, nxt, rwd, trm):

        # Get current and next values
        cval = self.v_net.values(obs)
        nval = self.v_net.values(nxt)

        # Compute advantages
        tgt, adv = self.retrn.compute(rwd, cval, nval, trm)

        return tgt, adv

    # Prepare training data
    def prepare_data(self, size):

        self.data = self.gbuff.get_buffers(self.gnames, size)
        lgt = len(self.data["obs"])

        return lgt, True

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

        self.p_net.reset()
        self.v_net.reset()
        self.buff.reset()
        self.gbuff.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(["obs", "nxt", "act", "rwd", "trm"],
                        [ obs,   nxt,   act,   rwd,   trm ])
        
        # Store in SRL buffer
        self.srl.gbuff.store(["obs"],obs)
        # Update SRL counter
        self.srl.counter += 1

    # Actions to execute before the inner training loop
    def pre_loop(self):

        self.buff.reset()

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
        gtgt, gadv = self.returns(gobs, gnxt, grwd, gtrm)

        self.gbuff.store(self.gnames, [gobs, gact, gadv, gtgt, glgp])

    # Save agent parameters
    def save(self, filename):

        self.p_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.p_net.load(filename)
