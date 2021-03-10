# Generic imports
import math
import numpy as np

# Custom imports
from dragonfly.core.actor    import *
from dragonfly.core.critic   import *
from dragonfly.core.buff     import *
from dragonfly.core.report   import *
from dragonfly.core.adv      import *
from dragonfly.core.renderer import *
from dragonfly.core.counter  import *

###############################################
### A discrete PPO agent
class ppo:
    def __init__(self, act_dim, obs_dim, params):

        # Initialize from arguments
        self.name         = 'ppo'
        self.act_dim      = act_dim
        self.obs_dim      = obs_dim
        self.n_cpu        = params.n_cpu
        self.ep_end       = params.ep_end

        self.n_buff       = params.n_buff
        self.buff_size    = params.buff_size
        self.batch_frac   = params.batch_frac
        self.n_epochs     = params.n_epochs

        self.pol_clip     = params.pol_clip
        self.adv_clip     = params.adv_clip
        self.bootstrap    = params.bootstrap
        self.entropy_coef = params.entropy
        self.gamma        = params.gamma
        self.gae_lambda   = params.gae_lambda
        self.norm_adv     = params.norm_adv

        # Build networks
        self.actor  = actor (act_dim  = self.act_dim,
                             obs_dim  = self.obs_dim,
                             arch     = params.actor_arch,
                             lr       = params.actor_lr,
                             grd_clip = params.grd_clip,
                             pol_type = "multinomial")
        self.critic = critic(obs_dim  = self.obs_dim,
                             arch     = params.critic_arch,
                             lr       = params.critic_lr)

        # Initialize buffers
        self.loc_buff = loc_buff(self.n_cpu, self.obs_dim, self.act_dim)
        self.glb_buff = glb_buff(self.n_cpu, self.obs_dim, self.act_dim)

        # Initialize learning data report
        self.report   = report()

        # Initialize renderer
        self.renderer = renderer(self.n_cpu, params.render_every)

        # Initialize counter
        self.counter  = counter(self.n_cpu, params.n_ep)

        # Initialize inner temporary buffer
        self.init_tmp_data()

    # Get actions
    def get_actions(self, observations):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act   = np.zeros([self.n_cpu, self.act_dim])

        # Loop over cpus
        for i in range(self.n_cpu):
            obs      = observations[i]
            act[i,:] = self.actor.get_action(obs)

        return act

    # Training
    def train(self):

        # Handle fixed-size buffer termination
        for cpu in range(self.n_cpu):
            if (self.loc_buff.trm.buff[cpu][-1] == 0):
                self.loc_buff.trm.buff[cpu][-1] = 2

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
            obs, act, adv, tgt = self.glb_buff.get(self.n_buff,
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

                self.train_actor(btc_obs, btc_adv, btc_act)
                self.train_critic(btc_obs, btc_tgt, size)

        # Update old networks
        self.actor.set_weights()

    # Handle termination
    def handle_term(self, done, ep_end):

        # "done" possibly contains signals from multiple parallel
        # environments. We assume it does and unroll it in a loop
        trm = np.array([self.n_cpu])

        # Loop over environments
        for i in range(self.n_cpu):
            ep_step = self.counter.ep_step[i]

            if (not self.bootstrap):
                if (not done[i]): trm[i] = 0
                if (    done[i]): trm[i] = 1
            if (    self.bootstrap):
                if (    done[i] and ep_step <  ep_end-1): trm[i] = 1
                if (    done[i] and ep_step >= ep_end-1): trm[i] = 2
                if (not done[i] and ep_step <  ep_end-1): trm[i] = 0
                if (not done[i] and ep_step >= ep_end-1):
                    trm[i]  = 2
                    done[i] = True

        return trm, done

    # Finish if some episodes are done
    def finish_episodes(self, path, done):

        # Loop over environments and finalize/reset
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.store_report(cpu)
                self.print_episode()
                self.finish_rendering(path, cpu)
                self.counter.reset_ep(cpu)

    # Printings at the end of an episode
    def print_episode(self):

        # No initial printing
        if (self.counter.ep == 0): return

        # Average and print
        avg = np.mean(self.report.score[-25:])
        avg = f"{avg:.3f}"
        end = '\n'
        if (self.counter.ep < self.counter.n_ep): end = '\r'
        print('# Ep #'+str(self.counter.ep)+', avg score = '+str(avg)+'      ', end=end)

    # Init temporary data
    def init_tmp_data(self):

        # These values are temporary storage for report struct
        self.actor_loss   = 0.0
        self.entropy      = 0.0
        self.actor_gnorm  = 0.0
        self.kl_div       = 0.0
        self.critic_loss  = 0.0
        self.critic_gnorm = 0.0

    ################################
    ### Actor/critic wrappings
    ################################

    # Training function for actor
    def train_actor(self, obs, adv, act):

        outputs          = self.actor.train(obs, adv, act,
                                            self.pol_clip,
                                            self.entropy_coef)
        self.actor_loss  = outputs[0]
        self.kl_div      = outputs[1]
        self.actor_gnorm = outputs[2]
        self.entropy     = outputs[3]

    # Training function for critic
    def train_critic(self, obs, tgt, size):

        outputs           = self.critic.train(obs, tgt, size)
        self.critic_loss  = outputs[0]
        self.critic_gnorm = outputs[1]

    ################################
    ### Local buffer wrappings
    ################################

    # Reset local buffer
    def reset_buff(self):

        self.loc_buff.reset()

    # Test buffer loop criterion
    def test_buff_loop(self):

        return (self.loc_buff.size < self.buff_size-1)

    # Store transition in local buffer
    def store_transition(self, obs, nxt, act, rwd, trm):

        self.loc_buff.store(obs, nxt, act, rwd, trm)

    ################################
    ### Report wrappings
    ################################

    # Store data in report
    def store_report(self, cpu):

        self.report.append(episode      = self.counter.ep,
                           score        = self.counter.score[cpu],
                           length       = self.counter.ep_step[cpu],
                           actor_loss   = self.actor_loss,
                           critic_loss  = self.critic_loss,
                           entropy      = self.entropy,
                           actor_gnorm  = self.actor_gnorm,
                           critic_gnorm = self.critic_gnorm,
                           kl_div       = self.kl_div,
                           actor_lr     = self.actor.get_lr(),
                           critic_lr    = self.critic.get_lr())

    # Write learning data report
    def write_report(self, path, run):

        # Set filename with method name and run number
        filename = path+'/'+self.name+'_'+str(run)+'.dat'
        self.report.write(filename)

    ################################
    ### Renderer wrappings
    ################################

    # Return rendering selection array
    def get_render_cpu(self):

        return self.renderer.render

    # Store one rendering step for all cpus
    def store_rendering(self, rnd):

        self.renderer.store(rnd)

    # Finish rendering process
    def finish_rendering(self, path, cpu):

        self.renderer.finish(path, self.counter.ep, cpu)

    ################################
    ### Counter wrappings
    ################################

    # Test episode loop criterion
    def test_ep_loop(self):

        return self.counter.test_ep_loop()

    # Update score
    def update_score(self, rwd):

        return self.counter.update_score(rwd)

    # Update step
    def update_step(self):

        return self.counter.update_step()
