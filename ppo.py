# Generic imports
import gym
import math
import numpy as np

# Custom imports
from agent import *
from buff  import *

###############################################
### A discrete PPO agent
class ppo_agent:
    def __init__(self, act_dim, obs_dim, params):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim

        self.actor_lr     = params.actor_lr
        self.critic_lr    = params.critic_lr
        self.buff_size    = params.buff_size
        self.batch_frac   = params.batch_frac
        self.n_epochs     = params.n_epochs
        self.n_buff       = params.n_buff
        self.pol_clip     = params.pol_clip
        self.grd_clip     = params.grd_clip
        self.adv_clip     = params.adv_clip
        self.bootstrap    = params.bootstrap
        self.entropy      = params.entropy
        self.gamma        = params.gamma
        self.gae_lambda   = params.gae_lambda
        self.actor_arch   = params.actor_arch
        self.critic_arch  = params.critic_arch
        self.ep_end       = params.ep_end
        self.n_cpu        = params.n_cpu

        # Build networks
        self.critic     = critic(self.critic_arch,
                                 self.critic_lr,
                                 self.grd_clip)
        self.actor      = actor (self.actor_arch,
                                 self.act_dim,
                                 self.actor_lr,
                                 self.grd_clip)
        self.old_actor  = actor (self.actor_arch,
                                 self.act_dim,
                                 self.actor_lr,
                                 self.grd_clip)

        # Init parameters
        dummy = self.critic   (tf.ones([1,self.obs_dim]))
        dummy = self.actor    (tf.ones([1,self.obs_dim]))
        dummy = self.old_actor(tf.ones([1,self.obs_dim]))
        self.old_actor.set_weights (self.actor.get_weights())

        # Init buffers
        self.loc_buff = loc_buff(self.n_cpu, self.obs_dim, self.act_dim)
        self.glb_buff = glb_buff(self.n_cpu, self.obs_dim, self.act_dim)

        # Arrays to store learning data
        self.ep      = np.array([], dtype=np.float32) # episode number
        self.score   = np.array([], dtype=np.float32) # episode reward
        self.stp     = np.array([], dtype=np.float32) # step    number
        self.ls_act  = np.array([], dtype=np.float32) # actor   loss
        self.ls_crt  = np.array([], dtype=np.float32) # critic  loss
        self.ent     = np.array([], dtype=np.float32) # entropy
        self.nrm_act = np.array([], dtype=np.float32) # actor  gradient norm
        self.nrm_crt = np.array([], dtype=np.float32) # critic gradient norm
        self.kl_div  = np.array([], dtype=np.float32) # approx. kl divergence
        self.lr      = np.array([], dtype=np.float32) # learning rate schedule
        self.length  = np.array([], dtype=np.uint16 ) # episode length

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

        # Handle fixed-size buffer termination
        for cpu in range(self.n_cpu):
            if (self.loc_buff.trm.buff[cpu][-1] == 0):
                self.loc_buff.trm.buff[cpu][-1] = 2

        # Retrieve serialized arrays
        obs, nxt, act, rwd, trm = self.loc_buff.serialize()

        # Get current and next values
        crt_val = np.array(self.critic(tf.cast(obs, dtype=tf.float32)))
        nxt_val = np.array(self.critic(tf.cast(nxt, dtype=tf.float32)))

        # Compute advantages
        tgt, adv = self.compute_adv(rwd, crt_val, nxt_val, trm)

        # Store in global buffers
        self.glb_buff.store(obs, adv, tgt, act)

        # Retrieve n_buff buffers from history
        lgt, obs, adv, tgt, act = self.get_buffers()

        # Handle insufficient history compared to batch_size
        batch_size = math.floor(self.batch_frac*lgt)

        # Retrieve learning rate
        lr = self.actor.opt._decayed_lr(tf.float32)

        # Save actor weights
        act_weights = self.actor.get_weights()

        # Train actor
        for epoch in range(self.n_epochs):

            # Randomize batch
            sample = np.arange(lgt)
            np.random.shuffle(sample)
            sample = sample[:batch_size]

            btc_obs = [obs[i] for i in sample]
            btc_adv = [adv[i] for i in sample]
            btc_tgt = [tgt[i] for i in sample]
            btc_act = [act[i] for i in sample]

            btc_obs = tf.reshape(tf.cast(btc_obs, tf.float32),
                                 [batch_size,self.obs_dim])
            btc_adv = tf.reshape(tf.cast(btc_adv, tf.float32),
                                 [batch_size])
            btc_tgt = tf.reshape(tf.cast(btc_tgt, tf.float32),
                                 [batch_size])
            btc_act = tf.reshape(tf.cast(btc_act, tf.float32),
                                 [batch_size,self.act_dim])

            # Train networks
            act_out = self.train_actor (btc_obs, btc_adv, btc_act)
            crt_out = self.train_critic(btc_obs, btc_tgt, batch_size)

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
        adv = (adv-np.mean(adv))/(np.std(adv) + 1.0e-5)

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

        return [loss, entropy, norm, kl]

    # Training function for critic
    @tf.function
    def train_critic(self, obs, tgt, btc):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(self.critic(obs))
            val  = tf.reshape(val, [btc])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            crt_var     = self.critic.trainable_variables
            grads       = tape.gradient(loss, crt_var)
            norm        = tf.linalg.global_norm(grads)
        self.critic.opt.apply_gradients(zip(grads,crt_var))

        return [loss, norm]

    # Store for printing
    def store_learning_data(self, ep, length, score, outputs):

        ls_act  = outputs[0]
        ent     = outputs[1]
        nrm_act = outputs[2]
        kl_div  = outputs[3]
        ls_crt  = outputs[4]
        nrm_crt = outputs[5]
        lr      = outputs[6]

        self.ep      = np.append(self.ep,      ep)
        self.score   = np.append(self.score,   score)
        self.ls_act  = np.append(self.ls_act,  ls_act)
        self.ls_crt  = np.append(self.ls_crt,  ls_crt)
        self.ent     = np.append(self.ent,     ent)
        self.nrm_act = np.append(self.nrm_act, nrm_act)
        self.nrm_crt = np.append(self.nrm_crt, nrm_crt)
        self.kl_div  = np.append(self.kl_div,  kl_div)
        self.lr      = np.append(self.lr,      lr)
        self.length  = np.append(self.length,  length)

    # Write learning data
    def write_learning_data(self):

        filename = 'ppo.dat'
        np.savetxt(filename, np.transpose([self.ep,      self.score,
                                           self.length,  self.ls_act,
                                           self.ls_crt,  self.ent,
                                           self.nrm_act, self.nrm_crt,
                                           self.kl_div,  self.lr]))

    # Get buffers
    def get_buffers(self):

        # Handle insufficient history
        n_buff = self.n_buff
        n_buff = int(min(n_buff, len(self.glb_buff.obs)//self.buff_size))
        length = n_buff*self.buff_size

        # Retrieve buffers
        obs    = self.glb_buff.obs[-length:]
        adv    = self.glb_buff.adv[-length:]
        tgt    = self.glb_buff.tgt[-length:]
        act    = self.glb_buff.act[-length:]

        return length, obs, adv, tgt, act

    # Test looping criterion
    def test_loop(self):

        return (not (self.loc_buff.size >= self.buff_size-1))

    # Handle termination state
    def handle_termination(self, done, ep_step, ep_end):

        if (not self.bootstrap):
            if (not done): term = 0
            if (    done): term = 1
        if (    self.bootstrap):
            if (    done and ep_step <  ep_end-1): term = 1
            if (    done and ep_step >= ep_end-1): term = 2
            if (not done and ep_step <  ep_end-1): term = 0
            if (not done and ep_step >= ep_end-1):
                term = 2
                done = True

        return term, done

    # Printings at the end of an episode
    def print_episode(self, ep, n_ep):

        if (ep == 0): return

        avg = np.mean(self.score[-25:])
        avg = f"{avg:.3f}"

        if (ep <  n_ep-1):
            print('# Ep #'+str(ep)+', avg score = '+str(avg), end='\r')
        if (ep == n_ep-1):
            print('# Ep #'+str(ep)+', avg score = '+str(avg), end='\n')

