# Generic imports
import numpy      as np

# Tensorflow imports
import tensorflow as tf

###############################################
### Multinomial policy class (discrete)
### act_dim  : size of action vector required by environment
class multinomial():
    def __init__(self, act_dim):

        # Dimension of policy output
        self.dim = 1*act_dim

    # Call policy
    def call(self, params):

        # Sanitize output
        policy       = tf.cast(params, tf.float64)
        policy, norm = tf.linalg.normalize(policy, ord=1)

        policy       = np.asarray(policy)[0]
        actions      = np.random.multinomial(1, policy)
        actions      = np.float32(actions)

        return actions
