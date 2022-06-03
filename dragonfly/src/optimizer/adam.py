# Tensorflow imports
import tensorflow                    as     tf
from   tensorflow.keras.optimizers   import Adam

###############################################
### Adam optimizer class
### lr        : learning rate
### grd_clip  : gradient clipping value
### grad_vars : trainable variables from the associated network
class adam():
    def __init__(self, pms, grad_vars):

        # Set default values
        self.lr       = 1.0e-3
        self.grd_clip = 0.5

        # Check inputs
        if hasattr(pms, "lr"):       self.lr       = pms.lr
        if hasattr(pms, "grd_clip"): self.grd_clip = pms.grd_clip

        # Initialize optimizer
        # A fake optimization step is applied so the saved
        # weights and config have the correct sizes
        self.opt = Adam(learning_rate = self.lr,
                        clipnorm      = self.grd_clip)
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.opt.apply_gradients(zip(zero_grads, grad_vars))

        # Save weights and config
        self.init_weights = self.opt.get_weights()
        self.config       = self.opt.get_config()

    # Get current learning rate
    def get_lr(self):

        return self.opt._decayed_lr(tf.float32)

    # Apply gradients
    def apply_grads(self, grds):

        self.opt.apply_gradients(grds)

    # Reset weights and config
    def reset(self):

        self.opt.set_weights(self.init_weights)
        self.opt.from_config(self.config)
