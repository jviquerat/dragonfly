# Generic imports
import numpy      as np

# Tensorflow imports
import tensorflow as tf

###############################################
### Policy class
### pol_type : policy type
### act_dim  : size of action vector required by environment
class policy():
    def __init__(self,
                 pol_type,
                 act_dim):

        # Fill structure
        self.pol_type = pol_type

        # Handle policy function
        if (pol_type == "multinomial"):
            self.call = self.multinomial
            self.dim  = 1*act_dim
        if (pol_type == "normal"):
            self.call = self.normal
            self.dim  = 2*act_dim

    # Multinomial policy (discrete)
    def multinomial(self, params):

        # Sanitize output
        policy       = tf.cast(params, tf.float64)
        policy, norm = tf.linalg.normalize(policy, ord=1)

        policy       = np.asarray(policy)[0]
        actions      = np.random.multinomial(1, policy)
        actions      = np.float32(actions)

        return actions

    # Normal law policy (continuous)
    def normal(self, params):

        print("Normal law policy not implemented yet")
        exit()
