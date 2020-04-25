# Generic imports
import os
import gym
import warnings
import numpy as np

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow               as tf
import tensorflow.keras         as tk
import tensorflow.keras.backend as kb
tf.logging.set_verbosity(tf.logging.FATAL)

###############################################
### Class ppo
### A standard PPO agent
class ppo:
    def __init__(self,
                 act_dim, obs_dim, n_episodes, n_steps,
                 learn_rate, buff_size, batch_size, n_epochs,
                 clip, entropy, gamma, gae_lambda, update_alpha):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim
        self.mu_dim       = act_dim
        self.sig_dim      = act_dim
        self.n_episodes   = n_episodes
        self.n_steps      = n_steps
        self.size         = n_steps*n_episodes

        self.learn_rate   = learn_rate
        self.buff_size    = buff_size
        self.batch_size   = batch_size
        self.n_epochs     = n_epochs
        self.clip         = clip
        self.entropy      = entropy
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.update_alpha = update_alpha

        # Build actors
        self.critic     = self.build_critic()
        self.actor      = self.build_actor()
        self.old_actor  = self.build_actor()
        self.old_actor.set_weights (self.actor.get_weights())

        # Generate dummy inputs for custom loss
        self.dummy_adv  = np.zeros((1, 1))
        self.dummy_pred = np.zeros((1, 2*self.act_dim))

        # Storing buffers
        self.idx      = 0

        self.obs      = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.act      = np.zeros((self.size, self.act_dim), dtype=np.float32)
        self.rwd      = np.zeros( self.size,                dtype=np.float32)
        self.val      = np.zeros( self.size,                dtype=np.float32)
        self.dlt      = np.zeros( self.size,                dtype=np.float32)
        self.drw      = np.zeros( self.size,                dtype=np.float32)
        self.adv      = np.zeros( self.size,                dtype=np.float32)

        #self.gen      = np.zeros( self.size,                dtype=np.int32)
        #self.ep       = np.zeros( self.size,                dtype=np.int32)
        #self.bst_rwd  = np.zeros( self.n_gen,               dtype=np.float32)
        #self.bst_gen  = np.zeros( self.n_gen,               dtype=np.int32)
        #self.bst_ep   = np.zeros( self.n_gen,               dtype=np.int32)

    # Get batch of obs, actions and rewards
    def get_batch(self):

        # Start and end indices based on the required size of batch
        # Then draw a randomized version
        start = max(0,self.idx - self.batch_size)
        end   = self.idx
        rnd   = np.random.randint(start,end,self.batch_size)

        return self.obs[rnd,:], self.act[rnd,:], \
               self.adv[rnd],   self.mu [rnd,:], \
               self.sig[rnd,:], self.drw[rnd,:], \
               self.val[rnd,:]

    # Policy loss
    def policy_loss(self, adv, old_act):

        # Log prob density function
        def log_density(pred, y_true):

            # Compute log prob density
            mu      = pred[:, 0:self.act_dim]
            sig     = pred[:, self.act_dim:]
            var     = kb.square(sig)

            factor  = 1.0/kb.sqrt(2.*np.pi*var)
            pdf     = factor*kb.exp(-kb.square(y_true - mu)/(2.0*var))
            log_pdf = kb.log(pdf + kb.epsilon())

            return log_pdf

        def loss(y_true, y_pred):

            # Get the log prob density
            log_density_new = log_density(y_pred,  y_true)
            log_density_old = log_density(old_act, y_true)

            # Compute actor loss following Schulman
            ratio      = kb.exp(log_density_new - log_density_old)
            surrogate1 = ratio*adv

            clip_ratio = kb.clip(ratio,
                                 min_value = 1.0 - self.clip,
                                 max_value = 1.0 + self.clip)
            surrogate2 = clip_ratio*adv
            loss_actor =-kb.mean(tk.backend.minimum(surrogate1, surrogate2))

            # Compute entropy loss
            sig          = y_pred[:, self.act_dim:]
            var          = kb.square(sig)
            loss_entropy = kb.mean(-(kb.log(2.0*np.pi*var)+1.0)/2.0)

            # Total loss
            return loss_actor + self.entropy*loss_entropy

        return loss

    # Value loss
    def value_loss(self, old_val):

        def loss(y_true, y_pred):

            # Baseline mse
            surrogate1 = kb.square(y_pred - y_true)

            # Compute actor loss following Schulman
            clip_val   = kb.clip(y_pred,
                                 min_value = old_val - self.clip,
                                 max_value = old_val + self.clip)
            surrogate2 = kb.square(clip_val - y_true)

            return kb.mean(tk.backend.minimum(surrogate1, surrogate2))
        return loss

    # Build actor network using keras
    def build_actor(self):

        # Input layers
        # Forward network pass only requires observation
        # Advantage and old_action are only used in custom loss
        obs     = tk.layers.Input(shape=(self.obs_dim,)  )
        adv     = tk.layers.Input(shape=(1,),            )
        old_act = tk.layers.Input(shape=(2*self.act_dim,))

        # Use orthogonal layers initialization
        init_1  = tk.initializers.Orthogonal(gain=0.5, seed=None)
        init_2  = tk.initializers.Orthogonal(gain=0.1, seed=None)

        # Dense layer, then one branch for mu and one for sigma
        dense     = tk.layers.Dense(2,
                                   activation         = 'relu',
                                   kernel_initializer = init_1)(obs)
        mu        = tk.layers.Dense(self.act_dim,
                                   activation         = 'linear',
                                   kernel_initializer = init_2)(dense)
        sig       = tk.layers.Dense(self.act_dim,
                                   activation         = 'softplus',
                                   kernel_initializer = init_2)(dense)

        # Concatenate outputs
        outputs   = tk.layers.concatenate([mu, sig])

        # Generate actor
        actor     = tk.Model(inputs  = [obs, adv, old_act],
                             outputs = outputs)
        optimizer = tk.optimizers.Adam(lr = self.learn_rate)
        actor.compile(optimizer = optimizer,
                      loss      = self.policy_loss(adv, old_act))

        return actor

    # Build critic network using keras
    def build_critic(self):

        # Input layers
        obs     = tk.layers.Input(shape=(self.obs_dim,))
        old_val = tk.layers.Input(shape=(1,),          )

        # Use orthogonal layers initialization
        #init_1  = tk.initializers.Orthogonal(gain=0.5, seed=None)
        #init_2  = tk.initializers.Orthogonal(gain=0.1, seed=None)

        # Dense layer, then one branch for mu and one for sigma
        dense     = tk.layers.Dense(2,
                                    activation         = 'relu')(obs)
        value     = tk.layers.Dense(1,
                                    activation         = 'linear')(dense)

        # Generate actor
        critic    = tk.Model(inputs  = [obs, old_val],
                             outputs = value)
        optimizer = tk.optimizers.Adam(lr = self.learn_rate)
        crtic.compile(optimizer = optimizer,
                      loss      = self.value_loss(old_val))

        return critic

    # Copy new network weights to old one
    def update_old_network(self):

        # Actor
        actor_weights = self.actor.get_weights()
        self.old_actor.set_weights(actor_weights)

    # Get actions from network
    def get_actions(self, state):

        # Reshape state
        state   = state.reshape(1,self.obs_dim)

        # Predict means and deviations
        # The two last parameters are dummy arguments: they are
        # only required for the custom loss used for training
        outputs = self.actor.predict([state,self.dummy_adv,self.dummy_pred])
        mu      = outputs[0,            0:self.act_dim]
        sig     = outputs[0, self.act_dim:            ]

        # Draw action from normal law defined by mu and sigma
        actions = np.zeros(self.act_dim)

        for i in range(self.act_dim):
            actions[i] = np.random.normal(loc=mu[i], scale=sig[i])

        return actions, mu, sig

    # Get value from network
    def get_value(self, state):

        # Reshape state
        state = state.reshape(1,self.obs_dim)

        # Predict value
        val   = self.critic.predict([state,self.dummuy_adv])

        return val

    # Train networks
    def train_network(self):

        # Get batch
        obs, act, adv, mu, sig, drw, val = self.get_batch()

        # Compute old actions and values
        old_act = self.get_old_actions(obs)

        # Train networks
        self.actor.fit (x       = [obs, adv, old_act],
                        y       = act,
                        epochs  = self.n_epochs,
                        verbose = 0)
        self.critic.fit(x       = [obs, 
                        y       = drw,
                        epochs  = self.n_epochs,
                        verbose = 0)

        # Update old actor
        self.update_old_actor()

    # Store transitions into buffer
    def store_transition(self, obs, act, rwd, mu, sig):

        # Fill buffers
        self.obs [self.idx] = obs
        self.act [self.idx] = act
        self.rwd [self.idx] = rwd
        self.mu  [self.idx] = mu
        self.sig [self.idx] = sig

        # Update index
        self.idx           += 1

    # Compute advantages
    def compute_advantages(self):

        # Start and end indices of last generation
        start        = max(0,self.idx - self.n_ind)
        end          = self.idx

        # Compute normalized advantage
        avg_rwd      = np.mean(self.rwd[start:end])
        std_rwd      = np.std( self.rwd[start:end])
        self.adv[:]  = (self.rwd[:] - avg_rwd)/(std_rwd + 1.0e-7)

    # Compute delta
    def compute_delta(self, rwd, val, new_val):

        # Follow eq. (12) in PPO paper
        delta = rwd + self.gamma*new_val - val

        return delta

    # Get actions from previous policy
    def get_old_actions(self, state):

        # This is a batch of states, unlike in get_actions routine
        state   = state.reshape(min(self.n_ind,len(state)), self.obs_dim)
        outputs = self.old_actor.predict_on_batch([state,
                                                   self.dummy_adv,
                                                   self.dummy_pred])

        return outputs
