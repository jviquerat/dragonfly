# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Normal policy class with diagonal covariance matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_diag(base_normal):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.out_dim     = [self.dim, self.dim]
        self.store_dim   = self.act_dim
        self.store_type  = float
        self.target      = target

        self.sigma       = 1.0
        if (hasattr(pms, "sigma")): self.sigma = pms.sigma

        # Check parameters
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for mean of policy is not tanh")

        if (pms.network.heads.final[1] != "sigmoid"):
            warning("normal", "__init__",
                    "Final activation for stddev of policy is not sigmoid")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):

        mu, sg = self.forward(tf.cast(obs, tf.float32))
        act    = np.reshape(mu.numpy(), (-1,self.store_dim))

        return act

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        mu, sg = self.forward(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)

        return pdf

    # Networks forward pass
    @tf.function
    def forward(self, state):

        out = self.net.call(state)
        mu  = out[0]
        sg  = out[1]*self.sigma

        return mu, sg
