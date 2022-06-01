# Custom imports
from dragonfly.src.agent.base            import *
from dragonfly.src.terminator.terminator import *
from dragonfly.src.utils.buff            import *
from dragonfly.src.utils.counter         import *

###############################################
### DQN agent
class dqn():
    def __init__(self, obs_dim, act_dim, n_cpu, size, pms):

        # Initialize from arguments
        self.name        = 'dqn'
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.n_cpu       = n_cpu
        self.gamma       = pms.gamma
        self.mem_size    = size

        # Initialize random limit
        self.eps = decay_factory.create(pms.exploration.type,
                                        pms = pms.exploration)

        # Build values
        if (pms.value.type != "q_value"):
            error("dqn", "__init__",
                  "Value type for dqn agent is not q_value")

        self.q_net = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim,
                                        out_dim = act_dim,
                                        pms     = pms.value)

        # Create buffers
        self.buff = buff(self.n_cpu,
                        ["obs", "nxt", "act", "rwd", "dne", "stp", "trm"],
                        [obs_dim, obs_dim, 1, 1, 1, 1, 1])
        self.gbuff = gbuff(self.mem_size,
                           ["obs", "act", "tgt"],
                           [obs_dim, 1, 1])

        # Initialize counter
        self.counter = counter(self.n_cpu)

        # Initialize terminator
        self.terminator = terminator_factory.create(pms.terminator.type,
                                                    n_cpu = self.n_cpu,
                                                    pms   = pms.terminator)

    # Get actions
    def actions(self, obs):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, 1], dtype=int)

        for i in range(self.n_cpu):
            p = random.uniform(0, 1)
            if (p < self.eps.get()):
                act[i] = random.randrange(0, self.act_dim)
            else:
                val = self.q_net.values(tf.cast([obs[i]], tf.float32))
                act[i] = np.argmax(val)

        act = np.reshape(act, (-1))

        return act

    # Compute target
    def target(self, obs, nxt, act, rwd, trm):

        tgt = self.q_net.values(nxt)
        tgt = tf.reduce_max(tgt, axis=1)
        tgt = tf.reshape(tgt, [-1,1])
        tgt = rwd + trm*self.gamma*tgt

        return tgt

    # Prepare training data
    def prepare_data(self, size):

        names = ["obs", "act", "tgt"]
        self.data = self.gbuff.get_buffers(names, size)
        lgt = len(self.data["obs"])

        return lgt

    # Training
    def train(self, size):

        obs = self.data["obs"][:]
        act = self.data["act"][:]
        tgt = self.data["tgt"][:]

        tgt = tf.reshape(tgt, [size,-1])
        act = tf.reshape(act, [size,-1])
        act = tf.cast(act, tf.int32)

        self.q_net.loss.train(obs, act, tgt, self.q_net)

    # Reset
    def reset(self):

        self.counter.reset()
        self.q_net.reset()

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
        names = ["obs", "nxt", "act", "rwd", "trm"]
        data  = self.buff.serialize(names)
        gobs, gnxt, gact, grwd, gtrm = (data[name] for name in names)
        gtgt = self.target(gobs, gnxt, gact, grwd, gtrm)

        self.gbuff.store(["obs", "tgt", "act"],
                         [gobs,  gtgt,  gact ])

    # Save parameters
    def save(self, filename):

        self.q_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.q_net.load(filename)
