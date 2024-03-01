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
    def train(self, obs, adv, act, p, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            pdf     = p.compute_pdf(obs)
            lgp     = pdf.log_prob(act)
            loss_pg = tf.multiply(adv, lgp)
            loss_pg =-tf.reduce_mean(loss_pg)

            # Compute entropy loss
            entropy = tf.reshape(pdf.entropy(), [-1])
            entropy = tf.reduce_mean(entropy, axis=0)
            loss_entropy =-entropy

            # Compute total loss
            loss = loss_pg + self.ent_coef*loss_entropy

            # Apply gradients
            var   = p.net.trainables()
            grads = tape.gradient(loss, var)
        opt.apply_grads(zip(grads, var))
