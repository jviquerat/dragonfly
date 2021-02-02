# Generic imports
import numpy as np

###############################################
### Report buffer, used to store learning data
class report:
    def __init__(self):

        # Initialize empty arrays
        self.episode          = []
        self.step             = []
        self.score            = []
        self.length           = []
        self.actor_loss       = []
        self.critic_loss      = []
        self.entropy          = []
        self.actor_grad_norm  = []
        self.critic_grad_norm = []
        self.kl_divergence    = []
        self.actor_lr         = []
        self.critic_lr        = []

        # Initialize step
        self.step_count = 0

    # Append data to the report
    def append(self,
               episode,
               score            = None,
               length           = None,
               actor_loss       = None,
               critic_loss      = None,
               entropy          = None,
               actor_grad_norm  = None,
               critic_grad_norm = None,
               kl_divergence    = None,
               actor_lr         = None,
               critic_lr        = None):

        # Update step count if possible
        if (length is not None): self.step_count += length

        # Append data
        self.episode.append(episode)
        self.step.append(self.step_count)
        self.score.append(score)
        self.length.append(length)
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        self.entropy.append(entropy)
        self.actor_grad_norm.append(actor_grad_norm)
        self.critic_grad_norm.append(critic_grad_norm)
        self.kl_divergence.append(kl_divergence)
        self.actor_lr.append(actor_lr)
        self.critic_lr.append(critic_lr)

    # Write report
    def write(self, filename):

        # Generate array to save
        array = np.transpose([np.array(self.episode,          dtype=float),
                              np.array(self.score,            dtype=float),
                              np.array(self.length,           dtype=float),
                              np.array(self.actor_loss,       dtype=float),
                              np.array(self.critic_loss,      dtype=float),
                              np.array(self.entropy,          dtype=float),
                              np.array(self.actor_grad_norm,  dtype=float),
                              np.array(self.critic_grad_norm, dtype=float),
                              np.array(self.kl_divergence,    dtype=float),
                              np.array(self.actor_lr,         dtype=float),
                              np.array(self.critic_lr,        dtype=float),
                              np.array(self.step,             dtype=float)])
        array = np.nan_to_num(array, nan=0.0)

        # Save as a csv file
        np.savetxt(filename, array, fmt='%.5e')
