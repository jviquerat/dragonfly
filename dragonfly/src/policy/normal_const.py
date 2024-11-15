# Custom imports
from dragonfly.src.policy.tfd  import *
from dragonfly.src.policy.base import base_normal

###############################################
### Normal policy class with isotropic covariance matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_const(base_normal):
    def __init__(self, obs_dim, obs_shape, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.obs_shape   = obs_shape
        self.dim         = self.act_dim
        self.out_dim     = [self.dim]
        self.store_type  = float
        self.sigma       = pms.sigma
        self.target      = target

        # Check parameters
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for mean of policy is not tanh")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):

        mu  = self.forward(tf.cast(obs, tf.float32))
        act = np.reshape(mu.numpy(), (-1, self.act_dim))

        return act

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        mu    = self.forward(obs)
        sg    = tf.constant([[self.sigma]])
        sigma = tf.tile(sg,[1,self.dim])
        pdf   = tfd.MultivariateNormalDiag(loc        = mu,
                                           scale_diag = sigma)

        return pdf

    # Networks forward pass
    @tf.function
    def forward(self, state):

        return self.net.call(state)
