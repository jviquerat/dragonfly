# Tensorflow imports
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

###############################################
### Q-policy loss class for SAC policy update
class q_pol_sac():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, p, q1, q2, alpha):
        with tf.GradientTape() as tape:

            # Reparameterization trick
            mu, sg = p.forward(obs)
            nm     = tf.random.normal(tf.shape(mu), 0.0, 1.0)
            act    = mu + sg*nm

            # Real distribution
            dn  = tfd.MultivariateNormalDiag(loc        = mu,
                                             scale_diag = sg)
            lgp = dn.log_prob(act)
            lgp = tf.reshape(lgp, [-1,1])

            # Log-prob of reparameterized action
            act = tf.tanh(act)
            sth = tf.math.log(1.0 - tf.square(act))
            lgp = lgp - tf.reduce_sum(sth)

            # Compute loss
            cct  = tf.concat([obs,act], axis=-1)
            tgt1 = q1.forward(cct)
            tgt2 = q2.forward(cct)
            tgt  = tf.minimum(tgt1, tgt2)
            tgt  = tgt - alpha*lgp
            loss =-tf.reduce_mean(tgt)

            # Apply gradients
            val_var = p.trainables
            grads   = tape.gradient(loss, val_var)
        p.opt.apply_grads(zip(grads, val_var))
