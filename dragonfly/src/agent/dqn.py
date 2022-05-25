# Custom imports
from dragonfly.src.agent.base import *

###############################################
### DQN agent
class dqn():
    def __init__(self, obs_dim, act_dim, n_cpu, pms):

        # Initialize from arguments
        self.name        = 'dqn'
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.n_cpu       = n_cpu
        self.gamma       = pms.gamma

        # Initialize random limit
        self.eps = decay_factory.create(pms.exploration.type,
                                        pms = pms.exploration)

        # pol_dim is the true dimension of the action provided to the env
        # As dqn is only for discrete actions, it is set to 1
        self.pol_dim = 1

        # Build values
        if (pms.value.type != "q_value"):
            error("dqn", "__init__",
                  "Value type for dqn agent is not q_value")

        self.q_net = val_factory.create(pms.value.type,
                                        inp_dim = obs_dim,
                                        out_dim = act_dim,
                                        pms     = pms.value)

    # Get actions
    def actions(self, obs):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, self.pol_dim], dtype=int)

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

    # Training
    def train(self, obs, act, tgt, size):

        self.q_net.train(obs, act, tgt, size)

    # Reset
    def reset(self):

        self.q_net.reset()
