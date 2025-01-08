# Custom imports
from dragonfly.src.agent.base   import *
from dragonfly.src.utils.polyak import polyak

###############################################
### DDPG agent
class ddpg(base_agent_off_policy):
    def __init__(self, spaces, n_cpu, mem_size, pms):
        super().__init__(spaces)

        # Initialize from arguments
        self.name      = 'ddpg'
        self.spaces    = spaces
        self.n_cpu     = n_cpu
        self.mem_size  = mem_size
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

        self.p = pol_factory.create(pms.policy.type,
                                    obs_dim   = self.obs_dim(),
                                    obs_shape = self.obs_shape(),
                                    act_dim   = self.true_act_dim(),
                                    pms       = pms.policy,
                                    target    = True)

        # Build values
        if (pms.value.type != "q_value"):
            error("ddpg", "__init__",
                  "Value type for ddpg agent is not q_value")

        if (pms.value.loss.type != "mse_ddpg"):
            error("ddpg", "__init__",
                  "Loss type for ddpg agent is not mse_ddpg")

        self.q = val_factory.create(pms.value.type,
                                    inp_dim   = self.obs_dim() + self.true_act_dim(),
                                    inp_shape = None,
                                    out_dim   = 1,
                                    pms       = pms.value,
                                    target    = True)

        # Polyak averager
        self.polyak = polyak(self.rho)

        # Create buffers
        self.create_buffers()

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
            act   = np.random.uniform(-1.0, 1.0, (self.n_cpu, self.true_act_dim()))
        else:
            act   = self.p.actions(obs)
            noise = np.random.normal(0.0, self.sigma,
                                     (self.n_cpu, self.true_act_dim()))
            act  += noise
            act   = np.clip(act, -1.0, 1.0)
        act = act.astype(np.float32)

        self.step += 1

        # Check for NaNs
        if (np.isnan(act).any()):
            error("ddpg", "get_actions", "Detected NaN in generated actions")

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
        act  = self.p.reshape_actions(act)
        rwd  = tf.reshape(rwd, [size,-1])
        trm  = tf.reshape(trm, [size,-1])

        # Train q network
        self.q.loss.train(obs, nxt, act, rwd, trm,
                          self.gamma, self.p.net,
                          self.q.net, self.q.tgt, self.q.opt)

        # Train policy network
        self.p.loss.train(obs, self.p.net, self.q.net, self.p.opt)

        # Update target networks
        self.polyak.average(self.p.net, self.p.tgt)
        self.polyak.average(self.q.net, self.q.tgt)

    # Reset
    def reset(self):

        self.step = 0
        self.p.reset()
        self.q.reset()
        self.buff.reset()
        self.gbuff.reset()
