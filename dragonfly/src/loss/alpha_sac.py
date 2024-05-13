# Tensorflow imports
import tensorflow as tf

###############################################
### Alpha SAC loss
class alpha_sac():
    def __init__(self):
        pass

    # Train
    @tf.function
    def train(self, obs, p, log_alpha, tgt_entropy, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            act, lgp = p.sample(obs)
            loss     = log_alpha*(lgp + tgt_entropy)
            loss     =-tf.reduce_mean(loss)
            grads    = tape.gradient(loss, [log_alpha])

        opt.apply_grads(zip(grads, [log_alpha]))

