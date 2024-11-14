# Custom imports
from dragonfly.src.agent.base   import *
from dragonfly.src.utils.polyak import polyak

###############################################
### TD3 agent
class td3(base_agent_off_policy):
    def __init__(self, spaces, n_cpu, size, pms):
        super().__init__(spaces)

        # Initialize from arguments
        self.name       = 'td3'
        self.n_cpu      = n_cpu
        self.mem_size   = size
        self.gamma      = pms.gamma
        self.rho        = pms.rho
        self.n_warmup   = pms.n_warmup
        self.n_filling  = pms.n_filling
        self.p_update   = pms.p_update
        self.sigma_act  = pms.sigma_act
        self.sigma_tgt  = pms.sigma_tgt
        self.noise_clip = pms.noise_clip

        # Local variables
        self.step   = 0
        self.p_step = 0

        # Build policies
        if (pms.policy.type != "deterministic"):
            error("td3", "__init__",
                  "Policy type for td3 agent is not deterministic")
        if (pms.policy.loss.type != "q_pol"):
            error("td3", "__init__",
                  "Policy loss type for td3 agent is not q_pol")

        self.p = pol_factory.create(pms.policy.type,
                                    obs_dim   = self.obs_dim(),
                                    obs_shape = self.obs_shape(),
                                    act_dim   = self.act_dim(),
                                    pms       = pms.policy,
                                    target    = True)

        # Build values
        if (pms.value.type != "q_value"):
            error("td3", "__init__",
                  "Value type for td3 agent is not q_value")

        if (pms.value.loss.type != "mse_td3"):
            error("td3", "__init__",
                  "Loss type for td3 agent is not mse_td3")

        self.q1 = val_factory.create(pms.value.type,
                                     inp_dim = self.obs_dim() + self.act_dim(),
                                     inp_shape = None,
                                     out_dim = 1,
                                     pms     = pms.value,
                                     target  = True)
        self.q2 = val_factory.create(pms.value.type,
                                     inp_dim = self.obs_dim() + self.act_dim(),
                                     inp_shape = None,
                                     out_dim = 1,
                                     pms     = pms.value,
                                     target  = True)

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
            act   = np.random.uniform(-1.0, 1.0, (self.n_cpu, self.act_dim()))
        else:
            act   = self.p.actions(obs)
            noise = np.random.normal(0.0, self.sigma_act,
                                     (self.n_cpu, self.act_dim()))
            act  += noise
            act   = np.clip(act, -1.0, 1.0)
        act = act.astype(np.float32)

        self.step += 1

        # Check for NaNs
        if (np.isnan(act).any()):
            error("td3", "get_actions",
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
        act  = self.p.reshape_actions(act)
        rwd  = tf.reshape(rwd, [size,-1])
        trm  = tf.reshape(trm, [size,-1])

        # Train q network
        self.q1.loss.train(obs, nxt, act, rwd, trm, self.gamma,
                           self.sigma_tgt, self.noise_clip,
                           self.p.tgt, self.q1.net, self.q1.tgt,
                           self.q2.tgt, self.q1.opt)
        self.q2.loss.train(obs, nxt, act, rwd, trm, self.gamma,
                           self.sigma_tgt, self.noise_clip,
                           self.p.tgt, self.q2.net, self.q1.tgt,
                           self.q2.tgt, self.q2.opt)

        # Train policy network
        self.p_step += 1
        if (self.p_step == self.p_update):
            self.p_step = 0
            self.p.loss.train(obs, self.p.net, self.q1.net, self.p.opt)

            # Update target networks
            self.polyak.average(self.p.net,  self.p.tgt)
            self.polyak.average(self.q1.net, self.q1.tgt)
            self.polyak.average(self.q2.net, self.q2.tgt)

    # Reset
    def reset(self):

        self.step = 0
        self.p_step = 0
        self.p.reset()
        self.q1.reset()
        self.q2.reset()
        self.buff.reset()
        self.gbuff.reset()
