# Tensorflow imports
import tensorflow                    as     tf
from   tensorflow.keras.optimizers   import Nadam

###############################################
### Optimizer class
### lr       : learning rate
### grd_clip : gradient clipping value
class optimizer():
    def __init__(self, lr, grd_clip, grad_vars):

        # Handle arguments
        self.lr       = lr
        self.grd_clip = grd_clip

        #self.reset(lr, grd_clip, grd_vars)

        # Initialize optimizer
        self.opt = Nadam(lr       = lr,
                         clipnorm = grd_clip)
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.opt.apply_gradients(zip(zero_grads, grad_vars))

        # Save initial weights
        self.init_weights = self.opt.get_weights()
        self.config       = self.opt.get_config()

    # Get current learning rate
    def get_lr(self):

        return self.opt._decayed_lr(tf.float32)

    # Apply gradients
    def apply_grads(self, grds):

        self.opt.apply_gradients(grds)

    # Reset weights
    def reset(self, grad_vars):

        # Initialize optimizer
        #self.opt = Nadam(lr       = self.lr,
        #                 clipnorm = self.grd_clip)
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.opt.apply_gradients(zip(zero_grads, grad_vars))
        self.opt.set_weights(self.init_weights)
        self.opt.from_config(self.config)


        #self.opt.set_weights(self.init_weights)
