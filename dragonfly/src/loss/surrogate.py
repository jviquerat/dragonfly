# Tensorflow imports
import tensorflow as tf

# Custom imports
from dragonfly.src.core.constants import *

###############################################
### Surrogate loss class
class surrogate():
    def __init__(self, pms):

        # Set default values
        self.pol_clip   = 0.2
        self.ent_coef   = 0.01

        # Check inputs
        if hasattr(pms, "pol_clip"): self.pol_clip = pms.pol_clip
        if hasattr(pms, "ent_coef"): self.ent_coef = pms.ent_coef

    # Train
    @tf.function
    def train(self, obs, adv, act, plg, p):
        with tf.GradientTape() as tape:

            # Compute ratio of probabilities
            p.compute_pdf(obs)
            lgp   = p.log_prob(act)
            ratio = tf.exp(lgp - plg)

            # Compute actor loss
            p1 = tf.multiply(adv,ratio)
            p2 = tf.clip_by_value(ratio,
                                  1.0-self.pol_clip,
                                  1.0+self.pol_clip)
            p2 = tf.multiply(adv,p2)
            loss_surrogate =-tf.reduce_mean(tf.minimum(p1,p2))

            # Compute entropy loss
            loss_entropy =-p.entropy(lgp)

            # Compute total loss
            loss = loss_surrogate + self.ent_coef*loss_entropy

            # Apply gradients
            pol_var = p.trainables
            grads   = tape.gradient(loss, pol_var)
        p.opt.apply_grads(zip(grads, pol_var))
