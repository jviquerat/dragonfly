# Custom imports
from dragonfly.src.agent.base   import *
from dragonfly.src.utils.polyak import *

###############################################
### DDPG agent
class ddpg(base_agent):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name      = 'ddpg'
        self.act_dim   = act_dim
        self.obs_dim   = obs_dim
        self.n_cpu     = n_cpu
        self.mem_size  = size
        self.gamma     = pms.gamma
        self.rho       = pms.rho
        self.n_warmup  = pms.n_warmup
        self.n_filling = pms.n_filling
        self.sigma     = pms.sigma

        # Build policies
        if (pms.policy.type != "deterministic"):
            error("ddpg", "__init__",
                  "Policy type for ddpg agent is not deterministic")
        if (pms.policy.loss.type != "q_pol"):
            error("ddpg", "__init__",
                  "Policy loss type for ddpg agent is not q_pol")

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
        self.pol_dim = self.p_net.store_dim

        # Build values
        if (pms.value.type != "q_value"):
            error("ddpg", "__init__",
                  "Value type for ddpg agent is not q_value")

        if (pms.value.loss.type != "mse_ddpg"):
            error("ddpg", "__init__",
                  "Loss type for ddpg agent is not mse_ddpg")

        self.q_net = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim+act_dim,
                                        out_dim = 1,
                                        pms     = pms.value)
        self.q_tgt = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim+act_dim,
                                        out_dim = 1,
                                        pms     = pms.value)
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())

        # Polyak averager
        self.polyak = polyak(self.rho)

        # Create buffers
        self.names = ["obs", "nxt", "act", "rwd", "trm"]
        self.sizes = [obs_dim, obs_dim, self.pol_dim, 1, 1]
        self.buff  = buff(self.n_cpu, self.names, self.sizes)
        self.gbuff = gbuff(self.mem_size, self.names, self.sizes)

        # Initialize counter
        self.counter = counter(self.n_cpu)

        # Initialize termination
        self.term = termination_factory.create(pms.termination.type,
                                               n_cpu = self.n_cpu,
                                               pms   = pms.termination)

    # Get actions
    def actions(self, obs):

        if (self.counter.step < self.n_warmup):
            act   = np.random.uniform(-1.0, 1.0, (self.n_cpu, self.act_dim))
        else:
            act   = self.p_net.actions(obs)
            noise = np.random.normal(0.0, self.sigma,
                                     (self.n_cpu, self.act_dim))
            act  += noise
            act   = np.clip(act, -1.0, 1.0)
        act = act.astype(np.float32)

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ddpg", "get_actions",
                  "Detected NaN in generated actions")

        return act

    # Control (deterministic actions)
    def control(self, obs):

        return self.p_net.control(obs)

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < max(size, self.n_filling)): return lgt

        self.data = self.gbuff.get_batches(self.names, size)

    # Training
    def train(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < max(size, self.n_filling)): return lgt

        obs = self.data["obs"][:]
        nxt = self.data["nxt"][:]
        act = self.data["act"][:]
        rwd = self.data["rwd"][:]
        trm = self.data["trm"][:]

        act = tf.reshape(act, [size,-1])
        rwd = tf.reshape(rwd, [size,-1])
        trm = tf.reshape(trm, [size,-1])

        # Train q network
        self.q_net.loss.train(obs, nxt, act, rwd, trm,
                              self.gamma, self.p_tgt, self.q_net, self.q_tgt)

        # Train policy network
        self.p_net.loss.train(obs, self.p_net, self.q_net)

        # Update target networks
        self.polyak.average(self.p_net.net, self.p_tgt.net)
        self.polyak.average(self.q_net.net, self.q_tgt.net)

    # Reset
    def reset(self):

        self.counter.reset()
        self.p_net.reset()
        self.q_net.reset()
        self.p_tgt.reset()
        self.q_tgt.reset()
        self.p_tgt.net.set_weights(self.p_net.net.get_weights())
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())
        self.buff.reset()
        self.gbuff.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne):

        trm = self.term.terminate(dne, self.counter.ep_step)
        self.buff.store(self.names, [obs, nxt, act, rwd, trm])
        self.counter.update(rwd)

    # Actions to execute before the inner training loop
    def pre_loop(self):

        self.buff.reset()

    # Actions to execute after the inner training loop
    def post_loop(self):

        data = self.buff.serialize(self.names)
        gobs, gnxt, gact, grwd, gtrm = (data[name] for name in self.names)

        self.gbuff.store(self.names, [gobs, gnxt, gact, grwd, gtrm])

    # Save agent parameters
    def save(self, filename):

        self.p_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.p_net.load(filename)
