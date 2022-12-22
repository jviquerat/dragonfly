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
        mu, sg   = self.forward(obs)
        act      = mu + tf.random.normal(tf.shape(mu))*sg
        act      = tf.reshape(act, [-1,self.store_dim])
        tanh_act = tf.tanh(act)

        lgp =-0.5*(((act - mu)/(sg + 1.0e-8))**2 + 2.0*tf.math.log(sg) + tf.math.log(2.0*np.pi))
        lgp = tf.reduce_sum(lgp, axis=1)
        lgp = tf.reshape(lgp, [-1,1])

        # Log-prob of reparameterized action
        #sth = tf.math.log(1.0 - tf.square(tanh_act) + 1.0e-8)
        sth  = 2.0*(np.log(2.0) - act - tf.nn.softplus(-2.0*act)) # from openai implementation
        sth  = tf.reduce_sum(sth, axis=1)
        sth  = tf.reshape(sth, [-1,1])
        lgp -= sth

        return tanh_act, lgp
