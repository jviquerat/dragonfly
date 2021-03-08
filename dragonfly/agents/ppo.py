# Generic imports
import math
import numpy as np

# Custom imports
from dragonfly.core.actor  import *
from dragonfly.core.critic import *
from dragonfly.core.buff   import *
from dragonfly.core.report import *
from dragonfly.core.adv    import *

###############################################
### A discrete PPO agent
class ppo:
    def __init__(self, act_dim, obs_dim, params):

        # Initialize from arguments
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim

        self.name         = 'ppo'
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
        self.norm_adv     = params.norm_adv
        self.actor_arch   = params.actor_arch
        self.critic_arch  = params.critic_arch
        self.ep_end       = params.ep_end
        self.n_cpu        = params.n_cpu

        # Build networks
        self.actor  = actor (act_dim  = self.act_dim,
                             obs_dim  = self.obs_dim,
                             arch     = self.actor_arch,
                             lr       = self.actor_lr,
                             grd_clip = self.grd_clip,
                             pol_type = "multinomial")
        self.critic = critic(obs_dim  = self.obs_dim,
                             arch     = self.critic_arch,
                             lr       = self.critic_lr)

        # Initialize buffers
        self.loc_buff = loc_buff(self.n_cpu, self.obs_dim, self.act_dim)
        self.glb_buff = glb_buff(self.n_cpu, self.obs_dim, self.act_dim)

        # Initialize learning data report
        self.report = report()

    # Retrieve data in buffers
    def get_buffers(self, n_buff, buff_size):

        end    = len(self.glb_buff.obs)
        start  = max(0,end - n_buff*buff_size)
        size   = end - start

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Retrieve buffers
        obs = [self.glb_buff.obs[i] for i in sample]
        act = [self.glb_buff.act[i] for i in sample]
        adv = [self.glb_buff.adv[i] for i in sample]
        tgt = [self.glb_buff.tgt[i] for i in sample]

        # Reshape
        obs = tf.reshape(tf.cast(obs, tf.float32), [size, self.obs_dim])
        act = tf.reshape(tf.cast(act, tf.float32), [size, self.act_dim])
        adv = tf.reshape(tf.cast(adv, tf.float32), [size])
        tgt = tf.reshape(tf.cast(tgt, tf.float32), [size])

        return obs, act, adv, tgt

    # Train networks
    def train_networks(self):

        # Handle fixed-size buffer termination
        for cpu in range(self.n_cpu):
            if (self.loc_buff.trm.buff[cpu][-1] == 0):
                self.loc_buff.trm.buff[cpu][-1] = 2

        # Retrieve learning rate
        lr = self.actor.save_lr()

        # Save actor weights
        self.actor.save_weights()

        # Retrieve serialized arrays
        obs, nxt, act, rwd, trm = self.loc_buff.serialize()

        # Get current and next values
        crt_val = self.critic.get_value(obs)
        nxt_val = self.critic.get_value(nxt)

        # Compute advantages
        tgt, adv = compute_adv(rwd, crt_val, nxt_val, trm,
                               gamma      = self.gamma,
                               gae_lambda = self.gae_lambda,
                               norm_adv   = self.norm_adv,
                               adv_clip   = self.adv_clip)

        # Store in global buffers
        self.glb_buff.store(obs, adv, tgt, act)

        # Train actor and critic
        for epoch in range(self.n_epochs):

            # Retrieve data
            obs, act, adv, tgt = self.get_buffers(self.n_buff,
                                                  self.buff_size)
            lgt      = self.n_buff*self.buff_size
            btc_size = math.floor(self.batch_frac*lgt)
            done     = False
            btc      = 0

            # Visit all available history
            while not done:

                start    = btc*btc_size
                end      = min((btc+1)*btc_size,len(obs))
                size     = end - start
                btc     += 1
                if (end  == len(obs)): done = True

                btc_obs  = obs[start:end]
                btc_act  = act[start:end]
                btc_adv  = adv[start:end]
                btc_tgt  = tgt[start:end]

                act_out  = self.train_actor (btc_obs, btc_adv, btc_act)
                crt_out  = self.train_critic(btc_obs, btc_tgt, size)

        # Update old networks
        self.actor.set_weights()

        return act_out + crt_out + [lr]

    # Training function for actor
    @tf.function
    def train_actor(self, obs, adv, act):
        with tf.GradientTape() as tape:

            # Compute ratio of probabilities
            prv_pol  = tf.convert_to_tensor(self.actor.pnet.call(obs))
            pol      = tf.convert_to_tensor(self.actor.call(obs))
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
            act_var = self.actor.net.trainable_variables
            grads   = tape.gradient(loss, act_var)
            norm    = tf.linalg.global_norm(grads)
        self.actor.opt.apply_gradients(zip(grads,act_var))

        return [loss, entropy, norm, kl]

    # Training function for critic
    @tf.function
    def train_critic(self, obs, tgt, btc):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(self.critic.call(obs))
            val  = tf.reshape(val, [btc])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            crt_var     = self.critic.net.trainable_variables
            grads       = tape.gradient(loss, crt_var)
            norm        = tf.linalg.global_norm(grads)
        self.critic.opt.apply_gradients(zip(grads,crt_var))

        return [loss, norm]

    # Write learning data report
    def write_report(self, path, run):

        # Set filename with method name and run number
        filename = path+'/'+self.name+'_'+str(run)+'.dat'
        self.report.write(filename)

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

        avg = np.mean(self.report.score[-25:])
        avg = f"{avg:.3f}"
        end = '\n'
        if (ep < n_ep): end = '\r'
        print('# Ep #'+str(ep)+', avg score = '+str(avg)+'      ',
              end=end)
