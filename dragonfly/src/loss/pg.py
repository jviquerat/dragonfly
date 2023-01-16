# Tensorflow imports
import tensorflow as tf

###############################################
### PG loss class
class pg():
    def __init__(self, pms):

        # Set default values
        self.ent_coef   = 0.01

        # Check inputs
        if hasattr(pms, "ent_coef"): self.ent_coef = pms.ent_coef

    # Train
    @tf.function
    def train(self, obs, adv, act, p):
        with tf.GradientTape() as tape:

            # Compute loss
            lgp     = p.log_prob(obs, act)
            lgp     = tf.multiply(adv, lgp)
            loss_pg =-tf.reduce_mean(lgp)

            # Compute entropy loss
            loss_entropy = tf.reduce_mean(-lgp)

            # Compute total loss
            loss = loss_pg + self.ent_coef*loss_entropy

            # Apply gradients
            pol_var = p.trainables
            grads   = tape.gradient(loss, pol_var)
        p.opt.apply_grads(zip(grads, pol_var))
