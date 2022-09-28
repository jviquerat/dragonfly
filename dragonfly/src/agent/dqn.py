# Custom imports
from dragonfly.src.agent.base import *

###############################################
### DQN agent
class dqn():
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name       = 'dqn'
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.n_cpu      = n_cpu
        self.mem_size   = size
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

        self.q_net = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim,
                                        out_dim = act_dim,
                                        pms     = pms.value)
        self.q_tgt = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim,
                                        out_dim = act_dim,
                                        pms     = pms.value)
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())

        # Create buffers
        self.names = ["obs", "nxt", "act", "rwd", "trm"]
        self.sizes = [obs_dim, obs_dim, 1, 1, 1]
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

        act = np.zeros([self.n_cpu, 1], dtype=int)

        for i in range(self.n_cpu):
            self.eps.decay()
            p = random.uniform(0, 1)
            if (p < self.eps.get()):
                act[i] = random.randrange(0, self.act_dim)
            else:
                cob = tf.cast([obs[i]], tf.float32)
                val = self.q_net.values(cob)
                act[i] = np.argmax(val)

        act = np.reshape(act, (-1))

        return act

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < size): return

        self.data = self.gbuff.get_batches(self.names, size)

    # Training
    def train(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < size): return lgt

        obs = self.data["obs"][:]
        nxt = self.data["nxt"][:]
        act = self.data["act"][:]
        rwd = self.data["rwd"][:]
        trm = self.data["trm"][:]

        act = tf.reshape(act, [size,-1])
        rwd = tf.reshape(rwd, [size,-1])
        trm = tf.reshape(trm, [size,-1])
        act = tf.cast(act, tf.int32)

        self.q_net.loss.train(obs, nxt, act, rwd, trm,
                              self.gamma, self.q_net, self.q_tgt)

        # Update target networks
        if (self.tgt_step == self.tgt_update):
            self.q_tgt.net.set_weights(self.q_net.net.get_weights())
            self.tgt_step  = 0
        else:
            self.tgt_step += 1

    # Reset
    def reset(self):

        self.tgt_step = 0
        self.counter.reset()
        self.q_net.reset()
        self.q_tgt.reset()
        self.q_tgt.net.set_weights(self.q_net.net.get_weights())
        self.buff.reset()
        self.gbuff.reset()
        self.eps.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
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

    # Save parameters
    def save(self, filename):

        self.q_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.q_net.load(filename)
