# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for DQN-style q networks
class mse_dqn():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, nxt, act, rwd, trm, gamma, q, qt):
        with tf.GradientTape() as tape:

            # Compute target
            tgt = qt.forward(nxt)
            tgt = tf.reduce_max(tgt, axis=1)
            tgt = tf.reshape(tgt, [-1,1])
            trm = tf.clip_by_value(trm, 0.0, 1.0)
            tgt = rwd + trm*gamma*tgt

            # Compute loss
            val  = tf.cast(q.forward(obs), tf.float32)
            val  = tf.reshape(val, [tf.size(tgt), -1])
            val  = tf.gather(val, act, axis=1, batch_dims=1)
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = q.trainables
            grads   = tape.gradient(loss, val_var)
        q.opt.apply_grads(zip(grads, val_var))
