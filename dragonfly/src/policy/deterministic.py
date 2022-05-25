# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Deterministic policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class deterministic(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = self.act_dim
        self.store_type = float
        self.kind       = "continuous"

        # Define and init network
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for network of deterministic policy is not tanh")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizers
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get actions
    def get_actions(self, obs):

        obs = tf.cast(obs, tf.float32)
        act = self.call_net(obs)

        noise = tf.random.normal([self.act_dim], 0, 1, tf.float32)
        act  += noise
        act   = np.reshape(act.numpy(), (self.store_dim))

        return act

    # Control (deterministic actions)
    #def control(self, obs):

        #obs    = tf.cast([obs], tf.float32)
        #mu, sg = self.call_net(obs)
        #act    = np.reshape(mu.numpy(), (self.store_dim))

    #    return act

    # Networks forward pass
    def call_net(self, state):

        out = self.net.call(state)

        return out

    # Compute policy entropy
    def entropy(self, obs):

        return [0.0]

    # Call loss for training
    def train(self, obs, adv, act, lgp):

        act = tf.reshape(act, [-1, self.act_dim])
        adv = tf.reshape(adv, [-1])
        lgp = tf.reshape(lgp, [-1])
        return self.loss.train(obs, adv, act, lgp, self)
