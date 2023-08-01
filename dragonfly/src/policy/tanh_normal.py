# Custom imports
from dragonfly.src.policy.base           import *
from dragonfly.src.policy.normal_log_std import *

###############################################
### Tanh-normal policy class (continuous)
### Inherits from normal class
class tanh_normal(normal):
    def __init__(self, obs_dim, act_dim, pms):

        super().__init__(obs_dim, act_dim, pms)

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        self.compute_pdf(obs)

        # Sample actions
        act      = self.pdf.sample(1)
        act      = tf.reshape(act, [-1,self.store_dim])
        tanh_act = tf.tanh(act)
        lgp      = self.log_prob(act)

        return tanh_act, lgp

    # Compute log_prob
    def log_prob(self, act):

        if (self.pdf is not None):
            lgp = self.pdf.log_prob(act)
            lgp = tf.reshape(lgp, [-1,1])

            # Compute log-prob of reparameterized action
            # Regular version, possibly numerically unstable
            # sth = tf.math.log(1.0 - tf.square(tanh_act) + 1.0e-8)

            # OpenAI version, numerically stable
            sth  = 2.0*(np.log(2.0) - act - tf.nn.softplus(-2.0*act))
            sth  = tf.reduce_sum(sth, axis=1)
            sth  = tf.reshape(sth, [-1,1])
            lgp -= sth

            return -lgp

    # Compute entropy
    def entropy(self, lgp):

        e = tf.reduce_mean(-lgp, axis=0)
        #print(e)
        return e
