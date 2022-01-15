# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class
class mse():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, tgt, btc, value):
        with tf.GradientTape() as tape:

            tgt = tf.reshape(tgt, [-1])

            # Compute loss
            val  = tf.convert_to_tensor(value.call_net(obs))
            val  = tf.reshape(val, [btc])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            val_var = value.trainables
            grads   = tape.gradient(loss, val_var)
        value.opt.apply_grads(zip(grads, val_var))
