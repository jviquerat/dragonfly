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
                 alg_type, act_dim, obs_dim, n_episodes,
                 learn_rate, buff_size, batch_size, n_epochs,
                 clip, entropy, gamma, gae_lambda, update_alpha):

        # Initialize from arguments
        self.alg_type     = alg_type
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
        if (self.alg_type == 'continuous'):
            self.actor      = self.build_continuous_actor()
            self.old_actor  = self.build_continuous_actor()
        if (self.alg_type == 'discrete'):
            self.actor      = self.build_discrete_actor()
            self.old_actor  = self.build_discrete_actor()
        self.old_actor.set_weights (self.actor.get_weights())

        # Generate dummy inputs for custom loss
        self.dummy_adv  = np.zeros((1, 1))
        if (self.alg_type == 'continuous'):
            self.dummy_pred = np.zeros((1, 2*self.act_dim))
        if (self.alg_type == 'discrete'):
            self.dummy_pred = np.zeros((1,   self.act_dim))

        # Storing buffers
        self.obs = cl.deque()
        self.act = cl.deque()
        self.rwd = cl.deque()
        self.val = cl.deque()
        self.dlt = cl.deque()
        self.tgt = cl.deque()
        self.adv = cl.deque()

    # Continuous policy loss
    def continuous_policy_loss(self, adv, old_act):
        def loss(y_true, y_pred):

            # New policy density
            new_mu    = y_pred[:,:self.act_dim ]
            new_sig   = y_pred[:, self.act_dim:]
            new_var   = kb.square(new_sig)
            new_den   = kb.sqrt(2.0*np.pi*new_var)
            new_prob  = kb.exp(-kb.square(y_true - new_mu)/(2.0*new_var))
            new_prob /= new_den

            # Old policy density
            old_mu    = old_act[:,:self.act_dim ]
            old_sig   = old_act[:, self.act_dim:]
            old_var   = kb.square(old_sig)
            old_den   = kb.sqrt(2.0*np.pi*old_var)
            old_prob  = kb.exp(-kb.square(y_true - old_mu)/(2.0*old_var))
            old_prob /= old_den

            # Compute loss
            ratio      = new_prob/(old_prob + kb.epsilon())
            surrogate1 = ratio*adv
            clip_ratio = kb.clip(ratio, 1.0-self.clip, 1.0+self.clip)
            surrogate2 = clip_ratio*adv
            loss_actor =-kb.mean(kb.minimum(surrogate1, surrogate2))

            # Compute entropy loss
            #loss_entropy = kb.mean((-kb.log(2.0*np.pi*new_var)+1.0)/2.0)
            loss_entropy = kb.mean(new_prob*kb.log(new_prob + kb.epsilon()))

            # Total loss
            return loss_actor + self.entropy*loss_entropy
        return loss

    # Discrete policy loss
    def discrete_policy_loss(self, adv, old_act):
        def loss(y_true, y_pred):

            new_prob = kb.sum(y_true*y_pred,  axis=-1)
            old_prob = kb.sum(y_true*old_act, axis=-1)
            ratio    = new_prob/(old_prob + kb.epsilon())

            surrogate1 = ratio*adv
            clip_ratio = kb.clip(ratio, 1.0-self.clip, 1.0+self.clip)
            surrogate2 = clip_ratio*adv
            loss_actor =-kb.mean(kb.minimum(surrogate1, surrogate2))

            # Compute entropy loss
            #loss_entropy = kb.mean((-kb.log(2.0*np.pi*new_var)+1.0)/2.0)
            loss_entropy = kb.mean(new_prob*kb.log(new_prob + kb.epsilon()))

            # Total loss
            return loss_actor + self.entropy*loss_entropy
        return loss

    # Build continuous actor network using keras
    def build_continuous_actor(self):

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
                      loss      = self.continuous_policy_loss(adv, old_act))

        return actor

    # Build discrete actor network using keras
    def build_discrete_actor(self):

        # Input layers
        # Forward network pass only requires observation
        # Advantage and old_action are only used in custom loss
        obs     = tk.layers.Input(shape=(self.obs_dim,)  )
        adv     = tk.layers.Input(shape=(1,),            )
        old_act = tk.layers.Input(shape=(self.act_dim,))

        # Use orthogonal layers initialization
        init_1  = tk.initializers.Orthogonal(gain=1.0, seed=None)
        init_2  = tk.initializers.Orthogonal(gain=1.0, seed=None)

        # Dense layer, then one branch for mu and one for sigma
        dense     = tk.layers.Dense(16,
                                   activation         = 'tanh',
                                   kernel_initializer = init_1)(obs)
        dense     = tk.layers.Dense(16,
                                   activation         = 'tanh',
                                   kernel_initializer = init_1)(dense)
        mu        = tk.layers.Dense(self.act_dim,
                                   activation         = 'softmax',
                                   kernel_initializer = init_2)(dense)

        # Generate actor
        actor     = tk.Model(inputs  = [obs, adv, old_act],
                             outputs = mu)
        optimizer = tk.optimizers.Adam(lr = self.learn_rate)
        actor.compile(optimizer = optimizer,
                      loss      = self.discrete_policy_loss(adv, old_act))

        return actor

    # Build critic network using keras
    def build_critic(self):

        # Input layers
        obs     = tk.layers.Input(shape=(self.obs_dim,)  )

        # Use orthogonal layers initialization
        init_1  = tk.initializers.Orthogonal(gain=0.1, seed=None)
        init_2  = tk.initializers.Orthogonal(gain=0.1, seed=None)

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

        if (self.alg_type == 'continuous'):
            mu      = outputs[0,            0:self.act_dim]
            sig     = outputs[0, self.act_dim:            ]

            # Draw action from normal law defined by mu and sigma
            actions = np.zeros(self.act_dim)

            for i in range(self.act_dim):
                actions[i] = np.random.normal(loc=mu[i], scale=sig[i])

        if (self.alg_type == 'discrete'):
            probs   = outputs[0,:]
            #print(probs)
            actions = np.random.multinomial(1, probs, size=1)
            actions = np.argmax(actions)
            #print(actions)

        return actions

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

        # Normalize advantages
        buff_adv[:] = (buff_adv[:] - np.mean(buff_adv))/np.std(buff_adv)

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
