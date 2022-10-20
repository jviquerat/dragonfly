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
    def train(self, obs, p, q):
        with tf.GradientTape() as tape:

            # Compute loss
            act  = p.forward(obs)
            cct  = tf.concat([obs,act], axis=-1)
            tgt  = q.forward(cct)
            loss =-tf.reduce_mean(tgt)

            # Apply gradients
            val_var = p.trainables
            grads   = tape.gradient(loss, val_var)
        p.opt.apply_grads(zip(grads, val_var))
