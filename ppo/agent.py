# Generic imports
import os
import warnings

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow                    as     tf
import tensorflow.keras              as     tk
import tensorflow_addons             as     tfa
import tensorflow_probability        as     tfp
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense
from   tensorflow.keras.initializers import Orthogonal

# Define alias
tfd = tfp.distributions

###############################################
### PPO actor
class actor(Model):
    def __init__(self, arch, act_dim, lr, grd_clip):
        super(actor, self).__init__()

        # Define network
        self.ac = []
        for layer in range(len(arch)):
            self.ac.append(Dense(arch[layer],
                                 kernel_initializer=Orthogonal(gain=1.0),
                                 activation = 'tanh'))
        self.ac.append(Dense(act_dim,
                             kernel_initializer=Orthogonal(gain=0.01),
                             activation = 'softmax'))

        # Define optimizer
        self.opt = tk.optimizers.Nadam(lr       = lr,
                                       clipnorm = grd_clip,
                                       beta_1   = 0.9,
                                       beta_2   = 0.999,
                                       epsilon  = 1.0e-5)

    # Network forward pass
    def call(self, state):

        # Copy inputs
        var = state

        # Compute output
        for layer in range(len(self.ac)):
            var = self.ac[layer](var)

        return var

###############################################
### PPO critic
class critic(Model):
    def __init__(self, arch, lr, grd_clip):
        super(critic, self).__init__()

        # Define network
        self.ct = []
        for layer in range(len(arch)):
            self.ct.append(Dense(arch[layer],
                                 kernel_initializer=Orthogonal(gain=1.0),
                                 activation = 'tanh'))
        self.ct.append(Dense(1,
                             kernel_initializer=Orthogonal(gain=1.0),
                             activation= 'linear'))

        # Define optimizer
        self.opt = tk.optimizers.Nadam(lr       = lr,
                                       beta_1   = 0.9,
                                       beta_2   = 0.999,
                                       epsilon  = 1.0e-5)

    # Network forward pass
    def call(self, state):

        # Copy input
        var = state

        # Compute output
        for layer in range(len(self.ct)):
            var = self.ct[layer](var)

        return var
