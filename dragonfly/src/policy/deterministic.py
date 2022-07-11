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
    def actions(self, obs):

        obs   = tf.cast(obs, tf.float32)
        act   = self.forward(obs)[0]
        act   = np.reshape(act.numpy(), (-1,self.store_dim))

        return act

    # Control (deterministic actions)
    def control(self, obs):

        obs = tf.cast(obs, tf.float32)
        act = self.forward(obs)[0]
        act = np.reshape(act.numpy(), (-1,self.store_dim))

        return act

    # Networks forward pass
    def forward(self, state):

        out = self.net.call(state)

        return out
