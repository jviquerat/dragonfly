# Custom imports
from dragonfly.src.agent.base import *

###############################################
### DQN agent
class dqn(base_agent_off_policy):
    def __init__(self, spaces, n_cpu, mem_size, pms):
        super().__init__(spaces)

        # Initialize from arguments
        self.name       = 'dqn'
        self.n_cpu      = n_cpu
        self.mem_size   = mem_size
        self.gamma      = pms.gamma
        self.tgt_update = pms.tgt_update

        # Local variables
        self.tgt_step   = 0

        # Initialize random limit
        self.eps = decay_factory.create(pms.exploration.type,
                                        pms = pms.exploration)

        # Build values
        if (pms.value.type != "q_value"):
            error("dqn", "__init__",
                  "Value type for dqn agent is not q_value")

        if (pms.value.loss.type != "mse_dqn"):
            error("dqn", "__init__",
                  "Loss type for dqn agent is not mse_dqn")

        pms.value.network.k_init       = "lecun_normal"
        pms.value.network.k_init_final = "lecun_normal"
        self.q = val_factory.create(pms.value.type,
                                    inp_dim   = self.obs_dim(),
                                    inp_shape = self.obs_shape(),
                                    out_dim   = self.natural_act_dim(),
                                    pms       = pms.value,
                                    target    = True)

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
        act = np.zeros([self.n_cpu, 1], dtype=int)

        for i in range(self.n_cpu):
            self.eps.decay()
            p = random.uniform(0, 1)
            if (p < self.eps.get()):
                act[i] = random.randrange(0, self.natural_act_dim())
            else:
                cob = tf.cast([obs[i]], tf.float32)
                val = self.q.values(cob)
                act[i] = np.argmax(val)

        act = np.reshape(act, (-1))
        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        val = self.q.values(obs)
        act = np.reshape(np.argmax(val), (-1))

        return act

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < size): return lgt, False

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
        act  = tf.reshape(act, [size,-1])
        rwd  = tf.reshape(rwd, [size,-1])
        trm  = tf.reshape(trm, [size,-1])
        act  = tf.cast(act, tf.int32)

        self.q.loss.train(obs, nxt, act, rwd, trm,
                          self.gamma, self.q.net,
                          self.q.net, self.q.opt)

        # Update target networks
        if (self.tgt_step == self.tgt_update):
            self.q.copy_tgt()
            self.tgt_step  = 0
        else:
            self.tgt_step += 1

    # Reset
    def reset(self):

        self.tgt_step = 0
        self.q.reset()
        self.buff.reset()
        self.gbuff.reset()
        self.eps.reset()

    # Save parameters
    # This function overrides the base_agent::save_policy
    def save_policy(self, filename):

        self.q.save(filename + '.weights.h5')

    # Load parameters
    # This function overrides the base_agent::load_policy
    def load_policy(self, folder):

        filename = folder + '/' + self.name + '.weights.h5'
        self.q.load(filename)
