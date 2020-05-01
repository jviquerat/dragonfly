# Generic imports
import os
import gym
import warnings
import numpy as np

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow               as     tf
import tensorflow.keras         as     tk
from   tensorflow.keras.layers  import Dense

###############################################
### A discrete PPO agent
class ppo_discrete:
    def __init__(self,
                 act_dim, obs_dim,
                 actor_lr, critic_lr, buff_size, batch_size, n_epochs,
                 l2_reg, orth_gain, clip, entropy, gamma, gae_lambda, alpha):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim
        self.mu_dim       = act_dim
        self.sig_dim      = act_dim

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
        self.step         = 0
        self.l2_reg       = l2_reg
        self.orth_gain    = orth_gain

        # Build actors
        self.critic    = self.build_critic()
        self.actor     = self.build_actor()
        self.old_actor = self.build_actor()
        self.old_actor.set_weights(self.actor.get_weights())

        # Generate dummy inputs for custom loss
        self.dum_adv  = np.zeros((1, 1))
        self.dum_pred = np.zeros((1, self.act_dim))

        # Storing for file outputs
        self.eps = [] # episode number
        self.rwd = [] # episode reward
        self.lgt = [] # episode length
        #self.val = [] # value
        #self.tgt = [] # target
        #self.adv = [] # advantage
        #self.stp = [] # step
        #self.ent = cl.deque() # entropy
        #self.als = cl.deque() # actor loss
        #self.cls = cl.deque() # critic loss

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
        obs     = tk.layers.Input(shape=(self.obs_dim,))
        adv     = tk.layers.Input(shape=(1,),          )
        old_pol = tk.layers.Input(shape=(self.act_dim,))

        # Use orthogonal layers initialization
        init   = tk.initializers.Orthogonal(gain=self.orth_gain)
        reg    = tk.regularizers.l2(self.l2_reg)

        # Dense layer, then one branch for mu and one for sigma
        dense  = Dense(64,
                       use_bias=False,
                       activation         = 'relu',
                       kernel_initializer = init,
                       kernel_regularizer=reg)(obs)
        dense  = Dense(64,
                       use_bias=False,
                       activation         = 'relu',
                       kernel_initializer = init,
                       kernel_regularizer=reg)(dense)
        pol    = Dense(self.act_dim,
                       activation         = 'softmax',
                       kernel_initializer = init,
                       kernel_regularizer=reg)(dense)

        # Generate actor
        actor = tk.Model(inputs = [obs, adv, old_pol], outputs = pol)

        return actor

    # Build critic network using keras
    def build_critic(self):

        # Input layers
        obs  = tk.layers.Input(shape=(self.obs_dim,))

        # Use orthogonal layers initialization
        init = tk.initializers.Orthogonal(gain=self.orth_gain)
        reg  = tk.regularizers.l2(self.l2_reg)

        # Dense layer, then one branch for mu and one for sigma
        dense = Dense(64,
                      use_bias=False,
                      kernel_initializer = init,
                      kernel_regularizer=reg,
                      activation = 'relu')(obs)
        value = Dense(1,
                      use_bias=False,
                      kernel_initializer = init,
                      kernel_regularizer=reg,
                      activation = 'linear')(dense)

        # Generate actor
        critic = tk.Model(inputs = obs, outputs = value)

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
        outputs = self.actor.predict([state,self.dum_adv,self.dum_pred])
        probs   = outputs[0,:]
        actions = np.random.multinomial(1, probs, size=1)
        actions = actions[0]

        return actions

    # Get value from network
    def get_value(self, state):

        # Reshape state
        state = state.reshape(1,self.obs_dim)

        # Predict value
        val   = self.critic.predict(state)
        val   = float(val)

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
                                      self.dum_adv,
                                      self.dum_pred], training=True)
                old_pol = self.old_actor([obs,
                                          self.dum_adv,
                                          self.dum_pred], training=True)
                loss    = self.policy_loss(act, adv, pol, old_pol)
                grads   = tape.gradient(loss, self.actor.trainable_variables)
                grads   = zip(grads, self.actor.trainable_variables)
                opt_actor.apply_gradients(grads)

            return loss

        @tf.function
        def train_critic(obs, tgt):
            with tf.GradientTape() as tape:
                val   = self.critic(obs, training=True)
                tgt   = tf.cast(tgt, tf.float32)
                loss  = tf.reduce_mean(tf.square(tgt - val))
                grads = tape.gradient(loss, self.critic.trainable_variables)
                grads = zip(grads, self.critic.trainable_variables)
                opt_critic.apply_gradients(grads)

            return loss

        # Train
        for epoch in range(self.n_epochs):
            #shuffle = np.random.randint(low  = 0,
            #                            high = self.buff_size,
            #                            size = self.batch_size)

            #print(obs[shuffle])
            #exit()
            loss_actor  = train_actor (obs, adv)
            loss_critic = train_critic(obs, tgt)

        # Update old actor
        self.update_old_actor()

    # Compute targets
    def compute_tgts(self, buff_rwd, buff_msk):

        # Initialize
        buff_tgt = np.zeros_like(buff_rwd)
        gm       = self.gamma
        t_t      = 0.0

        # Loop backward to compute discountet reward
        for r in reversed(range(len(buff_rwd))):
            m_t         = buff_msk[r]
            r_t         = buff_rwd[r]
            t_t         = r_t + gm*t_t*m_t
            buff_tgt[r] = t_t

        return buff_tgt

    # Compute deltas and advantages
    def compute_advs(self, buff_rwd, buff_val, buff_msk):

        # Initialize
        buff_adv = np.zeros_like(buff_rwd)
        buff_dlt = np.zeros_like(buff_rwd)
        gm       = self.gamma
        lbd      = self.gae_lambda

        # Compute deltas
        for t in range(len(buff_rwd)-1):
            m_t         = buff_msk[t]
            r_t         = buff_rwd[t]
            v_t         = buff_val[t]
            v_tp        = buff_val[t+1]
            buff_dlt[t] = r_t + gm*v_tp*m_t - v_t
        buff_dlt[-1]    = buff_rwd[-1] - buff_val[-1]

        # Compute advantages with GAE
        for t in range(len(buff_rwd)):
            # Loop from t to end of buffer and build a_t
            a_t = 0.0
            for l in range(t,len(buff_rwd)):
                d_l  = buff_dlt[l]
                a_t += d_l*(gm*lbd)**(l-t)
            buff_adv[t] = a_t

        # Normalize
        buff_adv = (buff_adv-np.mean(buff_adv))/(np.std(buff_adv) + 1.0e-10)

        return buff_adv
