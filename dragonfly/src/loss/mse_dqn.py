# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for DQN-style
class mse_dqn():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, act, tgt, btc, q):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(q.forward(obs))
            val  = tf.reshape(val, [btc, -1])
            val  = tf.gather(val, act, axis=1, batch_dims=1)
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = q.trainables
            grads   = tape.gradient(loss, val_var)
        q.opt.apply_grads(zip(grads, val_var))
