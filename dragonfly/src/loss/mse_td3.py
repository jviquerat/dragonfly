# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for TD3-style q networks
class mse_td3():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, nxt, act, rwd, trm, gamma, sigma, clp, pt, q, qt1, qt2):
        with tf.GradientTape() as tape:

            # Compute target
            nac  = pt.forward(nxt)
            nse  = tf.random.normal(tf.shape(nac), 0.0, sigma)
            nse  = tf.clip_by_value(nse, -clp, clp)
            nac  = tf.clip_by_value(nac+nse, -1.0, 1.0)
            nct  = tf.concat([nxt, nac], axis=-1)
            tgt1 = qt1.forward(nct)
            tgt2 = qt2.forward(nct)
            tgt  = tf.minimum(tgt1, tgt2)
            tgt  = tf.reshape(tgt, [-1,1])
            trm  = tf.clip_by_value(trm, 0.0, 1.0)
            tgt  = rwd + trm*gamma*tgt

            # Compute loss
            oac  = tf.concat([obs, act], axis=-1)
            val  = q.forward(oac)
            val  = tf.reshape(val, [-1,1])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = q.trainables
            grads   = tape.gradient(loss, val_var)
        q.opt.apply_grads(zip(grads, val_var))
