# Generic imports
import os
import gym
import warnings
import collections as cl
import numpy       as np

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
                 act_dim, obs_dim, n_episodes,
                 learn_rate, buff_size, batch_size, n_epochs,
                 clip, entropy, gamma, gae_lambda, update_alpha):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim
        self.mu_dim       = act_dim
        self.sig_dim      = act_dim
        self.n_episodes   = n_episodes

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
        self.obs = cl.deque()
        self.act = cl.deque()
        self.rwd = cl.deque()
        self.val = cl.deque()
        self.dlt = cl.deque()
        self.tgt = cl.deque()
        self.adv = cl.deque()

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
        init_1  = tk.initializers.Orthogonal(gain=1.0, seed=None)
        init_2  = tk.initializers.Orthogonal(gain=1.0, seed=None)

        # Dense layer, then one branch for mu and one for sigma
        dense     = tk.layers.Dense(16,
                                   activation         = 'tanh',
                                   kernel_initializer = init_1)(obs)
        #dense     = tk.layers.Dense(16,
        #                           activation         = 'tanh',
        #                           kernel_initializer = init_1)(dense)
        mu        = tk.layers.Dense(self.act_dim,
                                   activation         = 'tanh',
                                   kernel_initializer = init_2)(dense)
        sig       = tk.layers.Dense(self.act_dim,
                                   activation         = 'sigmoid',
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
        obs     = tk.layers.Input(shape=(self.obs_dim,)  )

        # Use orthogonal layers initialization
        init_1  = tk.initializers.Orthogonal(gain=1.0, seed=None)
        init_2  = tk.initializers.Orthogonal(gain=1.0, seed=None)

        # Dense layer, then one branch for mu and one for sigma
        dense     = tk.layers.Dense(16,
                                    activation = 'tanh',
                                    kernel_initializer = init_1)(obs)
        #dense     = tk.layers.Dense(16,
        #                           activation         = 'tanh',
        #                           kernel_initializer = init_1)(dense)
        value     = tk.layers.Dense(1,
                                    activation = 'linear',
                                    kernel_initializer = init_1)(dense)

        # Generate actor
        critic    = tk.Model(inputs  = obs,
                             outputs = value)
        optimizer = tk.optimizers.Adam(lr = self.learn_rate)
        critic.compile(optimizer = optimizer,
                      loss      = 'mean_squared_error')

        return critic

    # Copy new network weights to old one
    def update_old_actor(self):

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
        val   = self.critic.predict(state)

        return val

    # Train networks
    def train_networks(self, obs, act, adv, tgt):

        # Compute old actions and values
        old_act = self.get_old_actions(obs)

        # Train networks
        self.actor.fit (x          = [obs, adv, old_act],
                        y          = act,
                        epochs     = self.n_epochs,
                        batch_size = self.batch_size,
                        shuffle    = True,
                        verbose    = 0)
        self.critic.fit(x          = obs,
                        y          = tgt,
                        epochs     = self.n_epochs,
                        batch_size = self.batch_size,
                        shuffle    = True,
                        verbose    = 0)

        # Update old actor
        self.update_old_actor()

    # Compute delta
    # def compute_delta(self, rwd, val, new_val):

    #     # Follow eq. (12) in PPO paper
    #     delta = rwd + self.gamma*new_val - val

    #     return delta

    # Compute deltas, targets and advantages
    def compute_dlt_tgt_adv(self, buff_rwd, buff_tgt,
                                  buff_dlt, buff_adv,
                                  buff_val, buff_msk):

        prev_tgt = 0.0
        prev_val = 0.0
        prev_adv = 0.0
        coeff    = self.gamma*self.gae_lambda

        # Loop from the end of the buffer
        for i in reversed(range(self.buff_size)):

            # Get local variables
            # Mask is 0 when current state is the end of a trajectory
            rwd = buff_rwd[i]
            msk = buff_msk[i]
            val = buff_val[i]

            # V(s_t) = r_t + gamma*V(s_t+1)
            buff_tgt[i] = rwd + self.gamma*prev_tgt*msk

            # delta(s_t) = r_t + gamma*V(s_t+1) - V(s_t)
            buff_dlt[i] = rwd + self.gamma*prev_val*msk - val

            # A(s_t, a_t) = delta(s_t) + gamma*lamda*A(s_t+1, a_t+1)
            buff_adv[i] = buff_dlt[i] + coeff*prev_adv*msk

            # Update variables
            prev_tgt = buff_tgt[i]
            prev_val = val
            prev_adv = buff_adv[i]

        # Flip buffers
        #rev_rwd    = buff_rwd.copy()
        #rev_rwd[:] = np.flip(rev_rwd)[:]

        #rev_dlt    = buff_dlt.copy()
        #rev_dlt[:] = np.flip(rev_dlt)[:]        


        #rev_trm    = buff_trm.copy()
        #rev_trm[:] = np.flip(rev_trm)[:]

        # Compute target values using reversed reward buffer
        
        # tgt        = 0.0

        # for i in range(self.buff_size):
        #     # If this is terminal state, restart counting from 0
        #     if (rev_trm[i]): tgt = 0.0

        #     tgt         = rev_rwd[i] + self.gamma*tgt
        #     buff_tgt[i] = tgt

        # buff_tgt[:] = np.flip(buff_tgt)[:]




    # def compute_targets(self, buff_rwd, buff_tgt, buff_trm):

    #     # Compute target values using reversed reward buffer
    #     rev_rwd    = buff_rwd.copy()
    #     rev_rwd[:] = np.flip(rev_rwd)[:]
    #     rev_trm    = buff_trm.copy()
    #     rev_trm[:] = np.flip(rev_trm)[:]
    #     tgt        = 0.0

    #     for i in range(self.buff_size):
    #         # If this is terminal state, restart counting from 0
    #         if (rev_trm[i]): tgt = 0.0

    #         tgt         = rev_rwd[i] + self.gamma*tgt
    #         buff_tgt[i] = tgt

    #     buff_tgt[:] = np.flip(buff_tgt)[:]

    # # Compute advantages
    # def compute_advantages(self, buff_dlt, buff_adv, buff_trm):

    #     # Compute GAE using reversed delta buffer
    #     rev_dlt    = buff_dlt.copy()
    #     rev_dlt[:] = np.flip(rev_dlt)[:]
    #     rev_trm    = buff_trm.copy()
    #     rev_trm[:] = np.flip(rev_trm)[:]
    #     adv        = 0.0

    #     for i in range(self.buff_size):
    #         # If this is terminal state, restart counting from 0
    #         if (rev_trm[i]): adv = 0.0

    #         adv         = rev_dlt[i] + self.gamma*self.gae_lambda*adv
    #         buff_adv[i] = adv

    #     buff_adv[:] = np.flip(buff_adv)[:]

    #     # Normalize
    #     buff_adv = (buff_adv - np.mean(buff_adv))/np.std(buff_adv)

    # Store buffers
    def store_buffers(self, obs, act, rwd, val, dlt, tgt, adv):

        # Append to global buffers
        self.obs.append(obs)
        self.act.append(act)
        self.rwd.append(rwd)
        self.val.append(val)
        self.dlt.append(dlt)
        self.tgt.append(tgt)
        self.adv.append(adv)

    # Get actions from previous policy
    def get_old_actions(self, state):

        # This is a batch of states, unlike in get_actions routine
        state   = state.reshape(self.buff_size, self.obs_dim)
        outputs = self.old_actor.predict_on_batch([state,
                                                   self.dummy_adv,
                                                   self.dummy_pred])

        return outputs
