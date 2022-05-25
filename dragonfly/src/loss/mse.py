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

            # Compute loss
            val  = tf.convert_to_tensor(value.forward(obs))
            val  = tf.reshape(val, [btc])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = value.trainables
            grads   = tape.gradient(loss, val_var)
        value.opt.apply_grads(zip(grads, val_var))
