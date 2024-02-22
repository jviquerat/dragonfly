# Custom imports
from dragonfly.src.agent.base   import *
from dragonfly.src.utils.polyak import *

###############################################
### SAC agent
class sac(base_agent):
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name       = 'sac'
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.n_cpu      = n_cpu
        self.mem_size   = size
        self.gamma      = pms.gamma
        self.alpha      = pms.alpha
        self.rho        = pms.rho
        self.n_warmup   = pms.n_warmup
        self.n_filling  = pms.n_filling

        # Local variables
        self.step = 0

        # Build policies
        if (pms.policy.loss.type != "q_pol_sac"):
            error("sac", "__init__",
                  "Policy loss type for sac agent is not q_pol_sac")
        if (pms.policy.type != "tanh_normal"):
            error("sac", "__init__",
                  "Policy type for sac agent is not tanh_normal")

        self.p_net = pol_factory.create(pms.policy.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.policy)

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.pol_dim = self.p_net.store_dim

        # Build values
        if (pms.value.type != "q_value"):
            error("sac", "__init__",
                  "Value type for sac agent is not q_value")

        if (pms.value.loss.type != "mse_sac"):
            error("td3", "__init__",
                  "Loss type for sac agent is not mse_sac")

        self.q_net1 = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim+act_dim,
                                        out_dim = 1,
                                        pms     = pms.value)
        self.q_net2 = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim+act_dim,
                                        out_dim = 1,
                                        pms     = pms.value)
        self.q_tgt1 = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim+act_dim,
                                        out_dim = 1,
                                        pms     = pms.value)
        self.q_tgt2 = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim+act_dim,
                                        out_dim = 1,
                                        pms     = pms.value)
        self.q_tgt1.net.set_weights(self.q_net1.net.get_weights())
        self.q_tgt2.net.set_weights(self.q_net2.net.get_weights())

        # Polyak averager
        self.polyak = polyak(self.rho)

        # Create buffers
        self.names = ["obs", "nxt", "act", "rwd", "trm"]
        self.sizes = [obs_dim, obs_dim, self.pol_dim, 1, 1]
        self.buff  = buff(self.n_cpu, self.names, self.sizes)
        self.gbuff = gbuff(self.mem_size, self.names, self.sizes)

        # Initialize termination
        self.term = termination_factory.create(pms.termination.type,
                                               n_cpu = self.n_cpu,
                                               pms   = pms.termination)

        # Initialize timer
        self.timer_actions = timer("actions  ")

    # Get actions
    def actions(self, obs):

        self.timer_actions.tic()
        if (self.step < self.n_warmup):
            act = np.random.uniform(-1.0, 1.0, (self.n_cpu, self.act_dim))
            act = act.astype(np.float32)
        else:
            act, _ = self.p_net.actions(obs)

        self.step += 1

        # Check for NaNs
        if (np.isnan(act).any()):
            error("sac", "get_actions",
                  "Detected NaN in generated actions")
        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        return self.p_net.control(obs)

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < max(size, self.n_filling)): return lgt, False

        self.data = self.gbuff.get_batches(self.names, size)
        return lgt, True

    # Training
    def train(self, start, end):

        obs = self.data["obs"][start:end]
        nxt = self.data["nxt"][start:end]
        act = self.data["act"][start:end]
        rwd = self.data["rwd"][start:end]
        trm = self.data["trm"][start:end]

        size = end - start
        act  = self.p_net.reshape_actions(act)
        rwd  = tf.reshape(rwd, [size,-1])
        trm  = tf.reshape(trm, [size,-1])

        # Train q network
        self.q_net1.loss.train(obs, nxt, act, rwd, trm,
                               self.gamma, self.alpha, self.p_net,
                               self.q_net1, self.q_tgt1, self.q_tgt2)
        self.q_net2.loss.train(obs, nxt, act, rwd, trm,
                               self.gamma, self.alpha, self.p_net,
                               self.q_net2, self.q_tgt1, self.q_tgt2)

        # Train policy network
        self.p_net.loss.train(obs, self.p_net, self.q_net1, self.q_net2, self.alpha)

        # Update target networks
        self.polyak.average(self.q_net1.net, self.q_tgt1.net)
        self.polyak.average(self.q_net2.net, self.q_tgt2.net)

    # Reset
    def reset(self):

        self.step = 0
        self.p_net.reset()
        self.q_net1.reset()
        self.q_net2.reset()
        self.q_tgt1.reset()
        self.q_tgt2.reset()
        self.q_tgt1.net.set_weights(self.q_net1.net.get_weights())
        self.q_tgt2.net.set_weights(self.q_net2.net.get_weights())
        self.buff.reset()
        self.gbuff.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(self.names, [obs, nxt, act, rwd, trm])

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
