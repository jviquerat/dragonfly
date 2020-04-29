# Generic imports
import os
import gym
import warnings
import collections as cl
import numpy       as np

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow               as     tf
import tensorflow.keras         as     tk
import tensorflow.keras.backend as     kb
from   tensorflow.keras.layers  import Dense
#tf.logging.set_verbosity(tf.logging.FATAL)
#tf.keras.backend.set_floatx('float32')

###############################################
### A discrete PPO agent
class ppo_discrete:
    def __init__(self,
                 act_dim, obs_dim, n_episodes,
                 actor_lr, critic_lr, buff_size, batch_size, n_epochs,
                 clip, entropy, gamma, gae_lambda, alpha):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim
        self.mu_dim       = act_dim
        self.sig_dim      = act_dim
        self.n_episodes   = n_episodes

        self.actor_lr     = actor_lr
        self.critic_lr    = critic_lr
        self.buff_size    = buff_size
        self.batch_size   = batch_size
        self.n_epochs     = n_epochs
        self.clip         = clip
        self.entropy      = entropy
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.alpha        = alpha

        # Build actors
        self.critic    = self.build_critic()
        self.actor     = self.build_actor()
        self.old_actor = self.build_actor()
        self.old_actor.set_weights(self.actor.get_weights())

        # Generate dummy inputs for custom loss
        self.dummy_adv  = np.zeros((1, 1))
        self.dummy_pred = np.zeros((1, self.act_dim))

        # Storing buffers
        self.obs = cl.deque()
        self.act = cl.deque()
        self.rwd = cl.deque()
        self.val = cl.deque()
        self.dlt = cl.deque()
        self.tgt = cl.deque()
        self.adv = cl.deque()

    # Discrete policy loss
    def policy_loss(self, act, adv, pol, old_pol):

        # Compute ratio of probabilities
        new_prob = tf.reduce_sum(act*pol,     axis=1)
        old_prob = tf.reduce_sum(act*old_pol, axis=1)
        ratio    = new_prob/(old_prob + 1e-10)

        # Compute actor loss
        adv        = tf.cast(adv, tf.float32)
        p1         = tf.multiply(ratio,adv)
        p2         = tf.clip_by_value(ratio, 1.0-self.clip, 1.0+self.clip)
        p2         = tf.multiply(p2,adv)
        loss_actor =-tf.reduce_mean(tf.minimum(p1,p2))

        # Compute entropy loss
        entropy      =-(new_prob*tf.math.log(new_prob + 1.0e-10))
        loss_entropy =-tf.reduce_mean(entropy)

        # Compute total loss
        loss_total = loss_actor + self.entropy*loss_entropy

        return loss_total

    # Build discrete actor network
    def build_actor(self):

        # Input layers
        # Forward network pass only requires observation
        # Advantage and old_action are only used in custom loss
        obs        = tk.layers.Input(shape=(self.obs_dim,))
        adv        = tk.layers.Input(shape=(1,),          )
        old_policy = tk.layers.Input(shape=(self.act_dim,))

        # Use orthogonal layers initialization
        init   = tk.initializers.Orthogonal(gain=1.0)

        # Dense layer, then one branch for mu and one for sigma
        dense  = Dense(32,
                       use_bias=False,
                       activation         = 'relu',
                       kernel_initializer = init)(obs)
        dense  = Dense(32,
                       use_bias=False,
                       activation         = 'relu',
                       kernel_initializer = init)(dense)
        dense  = Dense(32,
                       use_bias=False,
                       activation         = 'relu',
                       kernel_initializer = init)(dense)
        policy = Dense(self.act_dim,
                       activation         = 'softmax',
                       kernel_initializer = init)(dense)

        # Generate actor
        actor = tk.Model(inputs  = [obs, adv, old_policy],
                         outputs = policy)

        return actor

    # Build critic network using keras
    def build_critic(self):

        # Input layers
        obs  = tk.layers.Input(shape=(self.obs_dim,))

        # Use orthogonal layers initialization
        init = tk.initializers.Orthogonal(gain=1.0)

        # Dense layer, then one branch for mu and one for sigma
        dense = Dense(32,
                      use_bias=False,
                      activation = 'relu',
                      kernel_initializer = init)(obs)
        dense = Dense(32,
                      use_bias=False,
                      activation = 'relu',
                      kernel_initializer = init)(dense)
        value = Dense(1,
                      use_bias=False,
                      activation = 'linear',
                      kernel_initializer = init)(dense)

        # Generate actor
        critic = tk.Model(inputs  = obs,
                          outputs = value)

        return critic

    # Update weights of old actor
    def update_old_actor(self):

        # Compute averaged weights
        new_w   = np.asarray(self.actor.get_weights())
        old_w   = np.asarray(self.old_actor.get_weights())
        weights = self.alpha*new_w + (1.0-self.alpha)*old_w
        self.old_actor.set_weights(weights)

    # Get actions from network
    def get_actions(self, state):

        # Reshape state
        state = state.reshape(1,self.obs_dim)

        # Predict means and deviations
        # The two last parameters are dummy arguments: they are
        # only required for the custom loss used for training
        outputs = self.actor.predict([state,self.dummy_adv,self.dummy_pred])
        probs   = outputs[0,:]
        actions = np.random.multinomial(1, probs, size=1)

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

        # Cast to proper tf types
        obs = tf.cast(obs, tf.float32)
        act = tf.cast(act, tf.float32)
        adv = tf.cast(adv, tf.float32)
        tgt = tf.cast(tgt, tf.float32)

        # Optimizers
        opt_actor  = tk.optimizers.Adam(lr=self.actor_lr)
        opt_critic = tk.optimizers.Adam(lr=self.critic_lr)

        # Training functions
        @tf.function
        def train_actor(obs, adv):
            with tf.GradientTape() as tape:
                pol     = self.actor([obs,
                                      self.dummy_adv,
                                      self.dummy_pred], training=True)
                old_pol = self.old_actor([obs,
                                          self.dummy_adv,
                                          self.dummy_pred], training=True)
                loss    = self.policy_loss(act, adv, pol, old_pol)
                grads   = tape.gradient(loss,self.actor.trainable_variables)
                grads   = zip(grads,self.actor.trainable_variables)
                opt_actor.apply_gradients(grads)

        @tf.function
        def train_critic(obs, tgt):
            with tf.GradientTape() as tape:
                val     = self.critic(obs, training=True)
                tgt     = tf.cast(tgt, tf.float32)
                loss    = -tf.reduce_mean(tf.square(tgt - val))
                grads   = tape.gradient(loss,self.critic.trainable_variables)
                grads   = zip(grads,self.critic.trainable_variables)
                opt_critic.apply_gradients(grads)

        # Train
        train_actor (obs, adv)
        train_critic(obs, tgt)

        # Update old actor
        self.update_old_actor()

    # Compute targets
    def compute_tgts(self, buff_rwd):

        # Initialize
        tgt      = 0.0
        buff_tgt = np.zeros_like(buff_rwd)

        # Loop backward to compute discountet reward
        for r in reversed(range(len(buff_rwd))):
            tgt         = buff_rwd[r] + self.gamma*tgt
            buff_tgt[r] = tgt

        return buff_tgt

    # Compute deltas and advantages
    def compute_advs(self, buff_rwd, buff_val):

        # Initialize
        buff_adv = np.zeros_like(buff_rwd)
        coeff    = self.gamma*self.gae_lambda

        for t in range(len(buff_rwd)):
            adv = 0.0
            for l in range(0, len(buff_rwd)-t-1):
                dlt = buff_rwd[t+l] + self.gamma*buff_val[t+l+1] - buff_val[t+l]
                adv += dlt*(self.gamma*self.gae_lambda)**l
            adv += (buff_rwd[t+l] - buff_val[t+l])*(self.gamma*self.gae_lambda)**l
            buff_adv[t] = adv
        buff_adv = (buff_adv-np.mean(buff_adv))/(np.std(buff_adv) + 1.0e-10)

        return buff_adv

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
