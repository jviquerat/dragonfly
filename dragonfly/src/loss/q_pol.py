# Tensorflow imports
import tensorflow as tf

import numpy as np

###############################################
### Q-policy loss class for DDPG policy update
class q_pol():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, p, q, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            lgt  = tf.shape(obs)[0]
            act  = tf.reshape(p.call(obs), [lgt,-1])
            cct  = tf.concat([obs,act], axis=-1)
            tgt  = tf.reshape(q.call(cct), [lgt,-1])
            loss =-tf.reduce_mean(tgt)

            # Apply gradients
            var   = p.trainables()
            grads = tape.gradient(loss, var)
        opt.apply_grads(zip(grads, var))
