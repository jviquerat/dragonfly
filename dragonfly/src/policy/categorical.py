# Custom imports
from dragonfly.src.policy.tfd  import *
from dragonfly.src.policy.base import base_policy

###############################################
### Categorical policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class categorical(base_policy):
    def __init__(self, obs_dim, obs_shape, act_dim, pms, target=False):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.obs_shape  = obs_shape
        self.out_dim    = [self.act_dim]
        self.store_type = int
        self.target     = target

        # Define and init network
        if (pms.network.heads.final[0] != "softmax"):
            warning("categorical", "__init__",
                    "Chosen final activation for categorical policy is not softmax")

        # Init from base class
        super().__init__(pms)

    # Get actions
    def actions(self, obs):

        act, lgp = self.sample(obs)

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):

        probs = self.forward(obs)
        act   = tf.argmax(probs[0][0])
        act   = tf.reshape(act, [-1])

        return act

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        pdf = self.compute_pdf(obs)

        # Sample actions
        act = tf.reshape(pdf.sample(1), [-1])
        lgp = tf.reshape(pdf.log_prob(act), [-1])

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):

        probs = self.forward(obs)
        return tfd.Categorical(probs=probs[0])

    # Network forward pass
    @tf.function
    def forward(self, obs):

        return self.net.call(obs)

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1])
