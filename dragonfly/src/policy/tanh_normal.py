# Custom imports
from dragonfly.src.policy.base   import *
from dragonfly.src.policy.normal import *

###############################################
### Tanh-normal policy class (continuous)
### Inherits from normal class
class tanh_normal(normal):
    def __init__(self, obs_dim, act_dim, pms):

        super().__init__(obs_dim, act_dim, pms)

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Reparameterization trick
        pdf = self.compute_pdf(obs)
        mu, sg = self.forward(obs)
        act = pdf.sample(1)
        act = tf.reshape(act, [-1,self.store_dim])
        lgp = pdf.log_prob(act)
        lgp = tf.reshape(lgp, [-1,1])
        tanh_act = tf.tanh(act)

        # Log-prob of reparameterized action
        sth = tf.math.log(1.0 - tf.square(tanh_act) + 1.0e-8)
        lgp = lgp - tf.reduce_sum(sth)

        return tanh_act, lgp
