# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for policy gradient-style value networks
class mse_pg():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, tgt, btc, v):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.cast(v.forward(obs), tf.float32)
            val  = tf.reshape(val, [btc])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = v.trainables
            grads   = tape.gradient(loss, val_var)
        v.opt.apply_grads(zip(grads, val_var))
