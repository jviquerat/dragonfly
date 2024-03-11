# Custom imports
from dragonfly.src.policy.base        import *
from dragonfly.src.policy.normal_diag import *

###############################################
### Tanh-normal policy class (continuous)
### Inherits from normal class
class tanh_normal(normal_diag):
    def __init__(self, obs_dim, act_dim, pms):

        super().__init__(obs_dim, act_dim, pms)

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Reparameterization trick
        mu, sg = self.forward(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)
        act = tf.reshape(pdf.sample(1), [-1,self.store_dim])
        lgp = tf.reshape(pdf.log_prob(act), [-1,1])

        tanh_act = tf.tanh(act)

        # Compute log-prob of reparameterized action
        # Regular version, possibly numerically unstable
        # sth = tf.math.log(1.0 - tf.square(tanh_act) + 1.0e-8)

        # OpenAI version, numerically stable
        sth  = 2.0*(np.log(2.0) - act - tf.nn.softplus(-2.0*act))
        sth  = tf.reduce_sum(sth, axis=1)
        sth  = tf.reshape(sth, [-1,1])
        lgp -= sth

        return tanh_act, lgp
