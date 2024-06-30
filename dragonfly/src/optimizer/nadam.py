# PyTorch imports
import torch
from torch.optim import NAdam

###############################################
### Nadam optimizer class
### lr        : learning rate
### grd_clip  : gradient clipping value
### grad_vars : trainable parameters from the associated network
class nadam():
    def __init__(self, pms, grad_vars):

        # Set default values
        self.lr       = 1.0e-3
        self.grd_clip = 0.5

        # Check inputs
        if hasattr(pms, "lr"):       self.lr       = pms.lr
        if hasattr(pms, "grd_clip"): self.grd_clip = pms.grd_clip

        # Initialize optimizer
        self.opt = NAdam(grad_vars, lr=self.lr)

        # Save initial state
        self.init_state = self.opt.state_dict()

    # Get current learning rate
    def get_lr(self):
        return self.opt.param_groups[0]['lr']
    
    def zero_grad(self):
        self.opt.zero_grad()

    # Apply gradients
    def apply_grads(self):
        # Clip gradients
        if self.grd_clip:
            torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]['params'], self.grd_clip)

        # Perform optimization step
        self.opt.step()

    # Reset weights and config
    def reset(self):
        self.opt.load_state_dict(self.init_state)
