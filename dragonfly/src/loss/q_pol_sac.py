# Tensorflow imports
import tensorflow as tf

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

            # Compute loss
            act, lgp  = p.sample(obs)
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
