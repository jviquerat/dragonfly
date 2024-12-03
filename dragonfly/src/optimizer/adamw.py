# Tensorflow imports
import tensorflow                    as     tf
from   tensorflow.keras.optimizers   import AdamW

###############################################
### AdamW optimizer class
class adamw():
    def __init__(self, pms, grad_vars):

        # Set default values
        self.lr       = 1.0e-3
        self.w_decay  = 1.0e-3
        self.epsilon  = 1.0e-7
        self.grd_clip = 0.5

        # Check inputs
        if hasattr(pms, "lr"):       self.lr       = pms.lr
        if hasattr(pms, "w_decay"):  self.w_decay  = pms.w_decay
        if hasattr(pms, "epsilon"):  self.epsilon  = pms.epsilon
        if hasattr(pms, "grd_clip"): self.grd_clip = pms.grd_clip

        # Initialize optimizer
        # A fake optimization step is applied so the saved
        # weights and config have the correct sizes
        self.opt = AdamW(learning_rate   = self.lr,
                         weight_decay    = self.w_decay,
                         epsilon         = self.epsilon,
                         global_clipnorm = self.grd_clip)
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.opt.apply_gradients(zip(zero_grads, grad_vars))

        # Save weights and config
        self.config       = self.opt.get_config()

    # Get current learning rate
    def get_lr(self):

        return self.opt._decayed_lr(tf.float32)

    # Apply gradients
    def apply_grads(self, grds):

        self.opt.apply_gradients(grds)

    # Reset weights and config
    def reset(self):

        self.opt.from_config(self.config)
