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
        self.episode       = []
        self.step          = []
        self.score         = []
        self.length        = []
        self.policy_loss   = []
        self.v_value_loss  = []
        self.entropy       = []
        self.policy_gnorm  = []
        self.v_value_gnorm = []
        self.kl_div        = []
        self.policy_lr     = []
        self.v_value_lr    = []

        # Initialize step
        self.step_count = 0

    # Append data to the report
    def append(self,
               episode       = None,
               score         = None,
               length        = None,
               policy_loss   = None,
               v_value_loss  = None,
               entropy       = None,
               policy_gnorm  = None,
               v_value_gnorm = None,
               kl_div        = None,
               policy_lr     = None,
               v_value_lr    = None):

        # Update step count if possible
        if (length is not None):
            self.step_count += length
            self.step.append(self.step_count)

        # Append data
        if (episode       is not None): self.episode.append(episode)
        if (score         is not None): self.score.append(score)
        if (length        is not None): self.length.append(length)
        if (policy_loss   is not None): self.policy_loss.append(policy_loss)
        if (v_value_loss  is not None): self.v_value_loss.append(v_value_loss)
        if (entropy       is not None): self.entropy.append(entropy)
        if (policy_gnorm  is not None): self.policy_gnorm.append(policy_gnorm)
        if (v_value_gnorm is not None): self.v_value_gnorm.append(v_value_gnorm)
        if (kl_div        is not None): self.kl_div.append(kl_div)
        if (policy_lr     is not None): self.policy_lr.append(policy_lr)
        if (v_value_lr    is not None): self.v_value_lr.append(v_value_lr)

    # Write report
    def write(self, filename):

        # Generate array to save
        array = np.transpose([np.array(self.episode,       dtype=float),
                              np.array(self.score,         dtype=float),
                              np.array(self.length,        dtype=float),
                              np.array(self.policy_loss,   dtype=float),
                              np.array(self.v_value_loss,  dtype=float),
                              np.array(self.entropy,       dtype=float),
                              np.array(self.policy_gnorm,  dtype=float),
                              np.array(self.v_value_gnorm, dtype=float),
                              np.array(self.kl_div,        dtype=float),
                              np.array(self.policy_lr,     dtype=float),
                              np.array(self.v_value_lr,    dtype=float),
                              np.array(self.step,          dtype=float)])
        array = np.nan_to_num(array, nan=0.0)

        # Save as a csv file
        np.savetxt(filename, array, fmt='%.5e')
