# Generic imports
import numpy as np

###############################################
### Report buffer, used to store learning data
class report:
    def __init__(self):

        self.reset()

    # Reset
    def reset(self):

        # Initialize empty arrays
        self.episode      = []
        self.step         = []
        self.score        = []
        self.length       = []
        self.actor_loss   = []
        self.critic_loss  = []
        self.entropy      = []
        self.actor_gnorm  = []
        self.critic_gnorm = []
        self.kl_div       = []
        self.actor_lr     = []
        self.critic_lr    = []

        # Initialize step
        self.step_count = 0

    # Append data to the report
    def append(self,
               episode      = None,
               score        = None,
               length       = None,
               actor_loss   = None,
               critic_loss  = None,
               entropy      = None,
               actor_gnorm  = None,
               critic_gnorm = None,
               kl_div       = None,
               actor_lr     = None,
               critic_lr    = None):

        # Update step count if possible
        if (length is not None):
            self.step_count += length
            self.step.append(self.step_count)

        # Append data
        if (episode      is not None): self.episode.append(episode)
        if (score        is not None): self.score.append(score)
        if (length       is not None): self.length.append(length)
        if (actor_loss   is not None): self.actor_loss.append(actor_loss)
        if (critic_loss  is not None): self.critic_loss.append(critic_loss)
        if (entropy      is not None): self.entropy.append(entropy)
        if (actor_gnorm  is not None): self.actor_gnorm.append(actor_gnorm)
        if (critic_gnorm is not None): self.critic_gnorm.append(critic_gnorm)
        if (kl_div       is not None): self.kl_div.append(kl_div)
        if (actor_lr     is not None): self.actor_lr.append(actor_lr)
        if (critic_lr    is not None): self.critic_lr.append(critic_lr)

    # Write report
    def write(self, filename):

        # Generate array to save
        array = np.transpose([np.array(self.episode,      dtype=float),
                              np.array(self.score,        dtype=float),
                              np.array(self.length,       dtype=float),
                              np.array(self.actor_loss,   dtype=float),
                              np.array(self.critic_loss,  dtype=float),
                              np.array(self.entropy,      dtype=float),
                              np.array(self.actor_gnorm,  dtype=float),
                              np.array(self.critic_gnorm, dtype=float),
                              np.array(self.kl_div,       dtype=float),
                              np.array(self.actor_lr,     dtype=float),
                              np.array(self.critic_lr,    dtype=float),
                              np.array(self.step,         dtype=float)])
        array = np.nan_to_num(array, nan=0.0)

        # Save as a csv file
        np.savetxt(filename, array, fmt='%.5e')
