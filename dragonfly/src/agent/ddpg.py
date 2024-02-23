# Custom imports
from dragonfly.src.agent.base   import *
from dragonfly.src.utils.polyak import *

###############################################
### DDPG agent
class ddpg(base_agent_off_policy):
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

        # Local variables
        self.step = 0

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
        self.create_buffers(act_dim=self.pol_dim)

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
            act   = np.random.uniform(-1.0, 1.0, (self.n_cpu, self.act_dim))
        else:
            act   = self.p_net.actions(obs)
            noise = np.random.normal(0.0, self.sigma,
                                     (self.n_cpu, self.act_dim))
            act  += noise
            act   = np.clip(act, -1.0, 1.0)
        act = act.astype(np.float32)

        self.step += 1

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ddpg", "get_actions",
                  "Detected NaN in generated actions")
        self.timer_actions.toc()

        return act

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
        self.q_net.loss.train(obs, nxt, act, rwd, trm,
                              self.gamma, self.p_tgt, self.q_net, self.q_tgt)

        # Train policy network
        self.p_net.loss.train(obs, self.p_net, self.q_net)

        # Update target networks
        self.polyak.average(self.p_net.net, self.p_tgt.net)
        self.polyak.average(self.q_net.net, self.q_tgt.net)

    # Reset
    def reset(self):

        self.step = 0
        self.p_net.reset()
        self.q_net.reset()
        self.p_tgt.reset()
        self.q_tgt.reset()
        self.p_tgt.net.set_weights(self.p_net.net.get_weights())
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())
        self.buff.reset()
        self.gbuff.reset()
