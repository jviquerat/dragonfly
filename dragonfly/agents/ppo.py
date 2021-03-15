# Generic imports
import math
import numpy as np

# Custom imports
from dragonfly.core.actor     import *
from dragonfly.core.critic    import *
from dragonfly.core.advantage import *
from dragonfly.utils.buff     import *
from dragonfly.utils.report   import *
from dragonfly.utils.renderer import *
from dragonfly.utils.counter  import *

###############################################
### PPO agent
class ppo():
    def __init__(self, act_dim, obs_dim, pms):

        # Initialize from arguments
        self.name         = 'ppo'
        self.n_vars       = 9

        self.act_dim      = act_dim
        self.obs_dim      = obs_dim
        self.n_cpu        = pms.n_cpu
        self.ep_end       = pms.ep_end

        self.n_buff       = pms.n_buff
        self.buff_size    = pms.buff_size
        self.btc_frac     = pms.batch_frac
        self.n_epochs     = pms.n_epochs

        self.pol_clip     = pms.pol_clip
        self.adv_clip     = pms.adv_clip
        self.bootstrap    = pms.bootstrap
        self.entropy_coef = pms.entropy
        self.gamma        = pms.gamma
        self.gae_lambda   = pms.gae_lambda
        self.adv_norm     = pms.adv_norm

        # Build networks
        self.actor  = actor (act_dim  = self.act_dim,
                             obs_dim  = self.obs_dim,
                             arch     = pms.actor_arch,
                             lr       = pms.actor_lr,
                             grd_clip = pms.grd_clip,
                             pol_type = pms.pol_type)
        self.critic = critic(obs_dim  = self.obs_dim,
                             arch     = pms.critic_arch,
                             lr       = pms.critic_lr)

        # Initialize buffers
        self.loc_buff = loc_buff(self.n_cpu,     self.obs_dim,
                                 self.act_dim,   self.buff_size)
        self.glb_buff = glb_buff(self.n_cpu,     self.obs_dim,
                                 self.act_dim,   self.n_buff,
                                 self.buff_size, self.btc_frac)

        # Initialize learning data report
        self.report   = report()

        # Initialize renderer
        self.renderer = renderer(self.n_cpu, pms.render_every)

        # Initialize counter
        self.counter  = counter(self.n_cpu, pms.n_ep)

        # Initialize inner temporary buffer
        self.init_tmp_data()

    # Reset
    def reset(self):
        self.actor.reset()
        self.critic.reset()
        self.loc_buff.reset()
        self.glb_buff.reset()
        self.report.reset()
        self.renderer.reset()
        self.counter.reset()

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

    # Finalize buffers before training
    def finalize_buffers(self):

        # Handle fixed-size buffer termination
        self.loc_buff.fix_trm_buffer()

        # Retrieve serialized arrays
        obs, nxt, act, rwd, trm = self.loc_buff.serialize()

        # Get current and next values
        crt_val = self.critic.get_value(obs)
        nxt_val = self.critic.get_value(nxt)

        # Compute advantages
        tgt, adv = advantage(rwd, crt_val, nxt_val, trm,
                             gamma      = self.gamma,
                             gae_lambda = self.gae_lambda,
                             adv_norm   = self.adv_norm,
                             adv_clip   = self.adv_clip)

        # Store in global buffers
        self.glb_buff.store(obs, adv, tgt, act)

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

    # Training
    def train(self):

        # Save actor weights
        self.actor.save_weights()

        # Train actor and critic
        for epoch in range(self.n_epochs):

            # Retrieve data
            obs, act, adv, tgt = self.glb_buff.get_buff()
            done               = False

            # Visit all available history
            while not done:
                start, end, done = self.glb_buff.get_indices()
                btc_obs          = obs[start:end]
                btc_act          = act[start:end]
                btc_adv          = adv[start:end]
                btc_tgt          = tgt[start:end]

                self.train_actor (btc_obs, btc_adv, btc_act)
                self.train_critic(btc_obs, btc_tgt, end - start)

        # Update old networks
        self.actor.set_weights()

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

        return self.loc_buff.test_buff_loop()

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
