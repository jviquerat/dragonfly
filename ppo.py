# Generic imports
import gym
import numpy as np

# Custom imports
from agent import *

###############################################
### A discrete PPO agent
class ppo_discrete:
    def __init__(self,
                 act_dim, obs_dim, actor_lr, critic_lr,
                 buff_size, batch_size, n_epochs, n_buff,
                 pol_clip, grd_clip, adv_clip, bootstrap,
                 entropy, gamma, gae_lambda, ep_end,
                 actor_arch, critic_arch, update_style):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim

        self.actor_lr     = actor_lr
        self.critic_lr    = critic_lr
        self.buff_size    = buff_size
        self.batch_size   = batch_size
        self.n_epochs     = n_epochs
        self.n_buff       = n_buff
        self.pol_clip     = pol_clip
        self.grd_clip     = grd_clip
        self.adv_clip     = adv_clip
        self.bootstrap    = bootstrap
        self.entropy      = entropy
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.actor_arch   = actor_arch
        self.critic_arch  = critic_arch
        self.update_style = update_style
        self.ep_end       = ep_end

        # Sanity check for batch_size
        if (batch_size > n_buff*buff_size):
            print('Error: batch_size too large')
            exit()

        # Build networks
        self.critic     = critic(critic_arch, critic_lr, grd_clip)
        self.actor      = actor (actor_arch, act_dim, actor_lr, grd_clip)
        self.old_actor  = actor (actor_arch, act_dim, actor_lr, grd_clip)

        # Init parameters
        dummy = self.critic   (tf.ones([1,self.obs_dim]))
        dummy = self.actor    (tf.ones([1,self.obs_dim]))
        dummy = self.old_actor(tf.ones([1,self.obs_dim]))
        self.old_actor.set_weights (self.actor.get_weights())

        # Local buffers
        self.reset_local_buffers()

        # Global buffers for off-policy training
        self.obs = np.empty((0,self.obs_dim))
        self.adv = np.empty((0,1))
        self.tgt = np.empty((0,1))
        self.act = np.empty((0,self.act_dim))

        # Arrays to store learning data
        self.ep      = np.array([]) # episode number
        self.score   = np.array([]) # episode reward
        self.stp     = np.array([]) # step    number
        self.ls_ppo  = np.array([]) # ppo     loss
        self.ls_act  = np.array([]) # actor   loss
        self.ls_crt  = np.array([]) # critic  loss
        self.ent     = np.array([]) # entropy
        self.nrm_act = np.array([]) # actor  gradient norm
        self.nrm_crt = np.array([]) # critic gradient norm
        self.kl_div  = np.array([]) # approx. kl divergence
        self.lr      = np.array([]) # learning rate schedule

    # Get actions from network
    def get_actions(self, state):

        # Reshape state
        state = tf.convert_to_tensor([state], dtype=tf.float32)

        # Forward pass to get policy
        policy  = self.actor(state)

        # Sanitize output
        policy       = tf.cast(policy, dtype=tf.float64)
        policy, norm = tf.linalg.normalize(policy, ord=1)

        policy  = np.asarray(policy)[0]
        actions = np.random.multinomial(1, policy)
        actions = np.float32(actions)

        return actions

    # Get value from network
    def get_value(self, state):

        # Reshape state
        state = state.reshape(1,self.obs_dim)

        # Predict value
        val   = self.critic(state)
        val   = float(val)

        return val

    # Train networks
    def train(self):

        # Reshape buffers
        self.reshape_local_buffers()

        obs = self.buff_obs
        nxt = self.buff_nxt
        act = self.buff_act
        rwd = self.buff_rwd
        trm = self.buff_trm

        # Get previous policy and values
        val = np.array(self.critic(tf.cast(obs, dtype=tf.float32)))
        nxt = np.array(self.critic(tf.cast(nxt, dtype=tf.float32)))

        # Compute advantages
        tgt, adv = self.compute_adv(rwd, val, nxt, trm)

        # Store in global buffers
        self.obs = np.append(self.obs, obs, axis=0)
        self.adv = np.append(self.adv, adv, axis=0)
        self.tgt = np.append(self.tgt, tgt, axis=0)
        self.act = np.append(self.act, act, axis=0)

        # Retrieve n_buff buffers from history
        n_buff = min(self.n_buff, len(self.obs)//self.buff_size)
        obs    = self.obs[-n_buff*self.buff_size:]
        adv    = self.adv[-n_buff*self.buff_size:]
        tgt    = self.tgt[-n_buff*self.buff_size:]
        act    = self.act[-n_buff*self.buff_size:]

        # Retrieve learning rate
        lr = self.actor.opt._decayed_lr(tf.float32)

        # Save actor weights
        act_weights = self.actor.get_weights()

        # Train actor
        for epoch in range(self.n_epochs):

            # Randomize batch
            sample = np.arange(n_buff*self.buff_size)
            np.random.shuffle(sample)
            sample = sample[:self.batch_size]

            btc_obs = [obs[i] for i in sample]
            btc_adv = [adv[i] for i in sample]
            btc_tgt = [tgt[i] for i in sample]
            btc_act = [act[i] for i in sample]

            btc_obs = tf.reshape(tf.cast(btc_obs, tf.float32),
                                 [self.batch_size,self.obs_dim])
            btc_adv = tf.reshape(tf.cast(btc_adv, tf.float32),
                                 [self.batch_size])
            btc_tgt = tf.reshape(tf.cast(btc_tgt, tf.float32),
                                 [self.batch_size])
            btc_act = tf.reshape(tf.cast(btc_act, tf.float32),
                                 [self.batch_size,self.act_dim])

            # Train networks
            act_out = self.train_actor (btc_obs, btc_adv, btc_act)
            crt_out = self.train_critic(btc_obs, btc_tgt)

        # Update old networks
        self.old_actor.set_weights(act_weights)

        return act_out + crt_out + [lr]

    # Compute deltas and advantages
    def compute_adv(self, rwd, val, nxt, trm):

        # Initialize
        gm  = self.gamma
        lbd = self.gae_lambda

        # Handle mask from termination signals
        msk = np.zeros(len(trm))
        for i in range(len(trm)):
            if (trm[i] == 0): msk[i] = 1.0
            if (trm[i] == 1): msk[i] = 0.0
            if (trm[i] == 2): msk[i] = 1.0

        # Compute deltas
        buff = zip(rwd, msk, nxt, val)
        dlt  = [r + gm*m*nv - v for r, m, nv, v in buff]
        dlt  = np.stack(dlt)

        # Compute advantages
        adv = dlt.copy()

        # Handle mask from termination signals
        msk = np.zeros(len(trm))
        for i in range(len(trm)):
            if (trm[i] == 0): msk[i] = 1.0
            if (trm[i] == 1): msk[i] = 0.0
            if (trm[i] == 2): msk[i] = 0.0

        for t in reversed(range(len(dlt)-1)):
            adv[t] += msk[t]*gm*lbd*adv[t+1]

        # Compute targets
        tgt  = adv.copy()
        tgt += val

        # Normalize
        adv = (adv-np.mean(adv))/(np.std(adv) + 1.0e-7)

        # Clip if required
        if (self.adv_clip):
            adv = np.maximum(adv, 0.0)

        return tgt, adv

    # Training function for actor
    @tf.function
    def train_actor(self, obs, adv, act):
        with tf.GradientTape() as tape:

            # Compute ratio of probabilities
            prv_pol  = tf.convert_to_tensor(self.old_actor(obs))
            pol      = tf.convert_to_tensor(self.actor(obs))
            new_prob = tf.reduce_sum(act*pol,     axis=1)
            prv_prob = tf.reduce_sum(act*prv_pol, axis=1)
            new_log  = tf.math.log(new_prob + 1.0e-5)
            old_log  = tf.math.log(prv_prob + 1.0e-5)
            ratio    = tf.exp(new_log - old_log)

            # Compute actor loss
            p1         = tf.multiply(adv,ratio)
            p2         = tf.clip_by_value(ratio,
                                          1.0-self.pol_clip,
                                          1.0+self.pol_clip)
            p2         = tf.multiply(adv,p2)
            loss_ppo   =-tf.reduce_mean(tf.minimum(p1,p2))

            # Compute entropy loss
            entropy      = tf.multiply(pol,tf.math.log(pol + 1.0e-5))
            entropy      =-tf.reduce_sum(entropy, axis=1)
            entropy      = tf.reduce_mean(entropy)
            loss_entropy =-entropy

            # Compute total loss
            loss = loss_ppo + self.entropy*loss_entropy

            # Compute KL div
            kl = tf.math.log(pol+1.0e-5) - tf.math.log(prv_pol+1.0e-5)
            kl = 0.5*tf.reduce_mean(tf.square(kl))

            # Apply gradients
            act_var = self.actor.trainable_variables
            grads   = tape.gradient(loss, act_var)
            norm    = tf.linalg.global_norm(grads)
        self.actor.opt.apply_gradients(zip(grads,act_var))

        return [loss_ppo, loss, entropy, norm, kl]

    # Training function for critic
    @tf.function
    def train_critic(self, obs, tgt):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(self.critic(obs))
            val  = tf.reshape(val,     [self.batch_size])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            crt_var     = self.critic.trainable_variables
            grads       = tape.gradient(loss, crt_var)
            norm        = tf.linalg.global_norm(grads)
        self.critic.opt.apply_gradients(zip(grads,crt_var))

        return [loss, norm]

    # Store for printing
    def store_learning_data(self, ep, score, outputs):

        ls_ppo  = outputs[0]
        ls_act  = outputs[1]
        ent     = outputs[2]
        nrm_act = outputs[3]
        kl_div  = outputs[4]
        ls_crt  = outputs[5]
        nrm_crt = outputs[6]
        lr      = outputs[7]

        self.ep      = np.append(self.ep,      ep)
        self.score   = np.append(self.score,   score)
        self.ls_ppo  = np.append(self.ls_ppo,  ls_ppo)
        self.ls_act  = np.append(self.ls_act,  ls_act)
        self.ls_crt  = np.append(self.ls_crt,  ls_crt)
        self.ent     = np.append(self.ent,     ent)
        self.nrm_act = np.append(self.nrm_act, nrm_act)
        self.nrm_crt = np.append(self.nrm_crt, nrm_crt)
        self.kl_div  = np.append(self.kl_div,  kl_div)
        self.lr      = np.append(self.lr,      lr)

    # Write learning data
    def write_learning_data(self):

        filename = 'ppo.dat'
        np.savetxt(filename, np.transpose([self.ep,      self.score,
                                           self.ls_ppo,  self.ls_act,
                                           self.ls_crt,  self.ent,
                                           self.nrm_act, self.nrm_crt,
                                           self.kl_div,  self.lr]))

    # Reset local buffers
    def reset_local_buffers(self):

        self.buff_obs = np.array([])
        self.buff_nxt = np.array([])
        self.buff_act = np.array([])
        self.buff_rwd = np.array([])
        self.buff_trm = np.array([])

    # Store transition in local buffers
    def store_transition(self, obs, nxt, act, rwd, trm):

        self.buff_obs = np.append(self.buff_obs, obs)
        self.buff_nxt = np.append(self.buff_nxt, nxt)
        self.buff_act = np.append(self.buff_act, act)
        self.buff_rwd = np.append(self.buff_rwd, rwd)
        self.buff_trm = np.append(self.buff_trm, trm)

    # Reshape local buffers
    def reshape_local_buffers(self):

        self.buff_obs = np.reshape(self.buff_obs,(self.buff_size,self.obs_dim))
        self.buff_nxt = np.reshape(self.buff_nxt,(self.buff_size,self.obs_dim))
        self.buff_act = np.reshape(self.buff_act,(self.buff_size,self.act_dim))
        self.buff_rwd = np.reshape(self.buff_rwd,(self.buff_size,1))
        self.buff_trm = np.reshape(self.buff_trm,(self.buff_size,1))

    # Test looping criterion
    def test_loop(self, ep_step, bf_step):

        if (self.update_style == 'ep'):
            if (ep_step == self.ep_end-1):
                return False
            else:
                return True

        if (self.update_style == 'buff'):
            if (bf_step == self.buff_size-1):
                return False
            else:
                return True
