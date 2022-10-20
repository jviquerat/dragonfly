# Custom imports
from dragonfly.src.policy.base   import *
from dragonfly.src.policy.normal import *

###############################################
### Tanh-normal policy class (continuous)
### Inherits from normal class
class tanh_normal(normal):
    def __init__(self, obs_dim, act_dim, pms):

        super().__init__(obs_dim, act_dim, pms)

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        mu, sg = self.forward(obs)
        nrm    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)
        pdf    = tfd.TransformedDistribution(nrm, tfp.bijectors.Tanh())

        return pdf
