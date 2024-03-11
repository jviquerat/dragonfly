# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Deterministic policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class deterministic(base_policy):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.out_dim    = [self.dim]
        self.store_dim  = self.act_dim
        self.store_type = float
        self.target     = target

        # Define and init network
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for network of deterministic policy is not tanh")

        # Init from base class
        super().__init__(pms)

    # Get actions
    def actions(self, obs):

        act   = self.forward(tf.cast(obs, tf.float32))
        act   = np.reshape(act.numpy(), (-1,self.store_dim))

        return act

    # Control (deterministic actions)
    def control(self, obs):

        return self.actions(obs)

    # Networks forward pass
    @tf.function
    def forward(self, state):

        out = self.net.call(state)[0]

        return out
