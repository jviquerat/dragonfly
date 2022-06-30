# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for DDPG-style q networks
class mse_ddpg():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, nxt, act, rwd, trm, gamma, pt, q, qt):
        with tf.GradientTape() as tape:

            # Compute target
            nac = pt.forward(nxt)[0]
            nct = tf.concat([nxt, nac], axis=-1)
            tgt = qt.forward(nct)[0]
            tgt = tf.reshape(tgt, [-1,1])
            trm = tf.clip_by_value(trm, 0.0, 1.0)
            tgt = rwd + trm*gamma*tgt

            # Compute loss
            oac  = tf.concat([obs, act], axis=-1)
            val  = tf.cast(q.forward(oac), tf.float32)
            val  = tf.reshape(val, [-1,1])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = q.trainables
            grads   = tape.gradient(loss, val_var)
        q.opt.apply_grads(zip(grads, val_var))
