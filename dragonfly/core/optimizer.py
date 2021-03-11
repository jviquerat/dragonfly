# Tensorflow imports
import tensorflow                    as     tf
from   tensorflow.keras.optimizers   import Nadam

###############################################
### Optimizer class
### lr       : learning rate
### grd_clip : gradient clipping value
class optimizer():
    def __init__(self, lr, grd_clip):

        # Handle arguments
        self.lr       = lr
        self.grd_clip = grd_clip

        # Initialize optimizer
        self.opt = Nadam(lr       = lr,
                         clipnorm = grd_clip)

        # Save initial weights
        self.init_weights = self.opt.get_weights()

    # Get current learning rate
    def get_lr(self):

        return self.opt._decayed_lr(tf.float32)

    # Apply gradients
    def apply_grads(self, grds):

        self.opt.apply_gradients(grds)

    # Reset weights
    def reset(self):

        pass
        #self.opt.set_weights(self.init_weights)
