# Generic imports
import math
import copy
import numpy as np

# Custom imports
from dragonfly.policy.policy  import *
from dragonfly.value.value    import *
from dragonfly.retrn.retrn    import *
from dragonfly.utils.buff     import *
from dragonfly.utils.report   import *
from dragonfly.utils.renderer import *
from dragonfly.utils.counter  import *

###############################################
### PPO agent
class ppo():
    def __init__(self, obs_dim, act_dim, pms):

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

        self.bootstrap    = pms.bootstrap

        # Build policies
        pms.policy.save      = True
        pms.policy.loss.type = "ppo"
        self.policy          = pol_factory.create(pms.policy.type,
                                                  obs_dim = obs_dim,
                                                  act_dim = act_dim,
                                                  pms     = pms.policy)

        # Build values
        pms.value.type = "v_value"
        self.v_value   = val_factory.create(pms.value.type,
                                            obs_dim = obs_dim,
                                            pms     = pms.value)

        # Build advantage
        self.retrn = retrn_factory.create(pms.retrn.type,
                                          pms = pms.retrn)

        # Initialize buffers
        self.loc_buff = loc_buff(self.n_cpu,     self.obs_dim,
                                 self.act_dim,   self.buff_size)
        self.glb_buff = glb_buff(self.n_cpu,     self.obs_dim,
                                 self.act_dim,   self.n_buff,
                                 self.buff_size, self.btc_frac)

        # Initialize learning data report
        self.report_fields = ["episode", "score", "length",
                              "policy_loss", "v_value_loss",
                              "entropy", "policy_gnorm",
                              "v_value_gnorm", "kl_div",
                              "policy_lr", "v_value_lr", "step"]
        self.report   = report(self.report_fields)

        # Initialize renderer
        self.renderer = renderer(self.n_cpu, pms.render_every)

        # Initialize counter
        self.counter  = counter(self.n_cpu, pms.n_ep)

        # Initialize inner temporary buffer
        self.init_tmp_data()

    # Reset
    def reset(self):
        self.policy.reset()
        self.v_value.reset()
        self.loc_buff.reset()
        self.glb_buff.reset()
        self.report.reset(self.report_fields)
        self.renderer.reset()
        self.counter.reset()

    # Get actions
    def get_actions(self, observations):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, self.act_dim])

        # Loop over cpus
        for i in range(self.n_cpu):
            obs      = observations[i]
            act[i,:] = self.policy.get_actions(obs)

        return act

    # Finalize buffers before training
    def finalize_buffers(self):

        # Handle fixed-size buffer termination
        self.loc_buff.fix_trm_buffer()

        # Retrieve serialized arrays
        obs, nxt, act, rwd, trm = self.loc_buff.serialize()

        # Get current and next values
        crt_val = self.v_value.get_values(obs)
        nxt_val = self.v_value.get_values(nxt)

        # Compute advantages
        tgt, adv = self.retrn.compute(rwd, crt_val, nxt_val, trm)

        # Store in global buffers
        self.glb_buff.store(obs, adv, tgt, act)

    # Handle termination
    def handle_term(self, done):

        # "done" possibly contains signals from multiple parallel
        # environments. We assume it does and unroll it in a loop
        trm = np.zeros([self.n_cpu])

        # Loop over environments
        for i in range(self.n_cpu):
            ep_step = self.counter.ep_step[i]

            if (not self.bootstrap):
                if (not done[i]): trm[i] = 0
                if (    done[i]): trm[i] = 1
            if (    self.bootstrap):
                if (    done[i] and ep_step <  self.ep_end-1): trm[i] = 1
                if (    done[i] and ep_step >= self.ep_end-1): trm[i] = 2
                if (not done[i] and ep_step <  self.ep_end-1): trm[i] = 0
                if (not done[i] and ep_step >= self.ep_end-1):
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
        if (self.counter.ep <= self.counter.n_ep):
            avg = np.mean(self.report.data["score"][-25:])
            avg = f"{avg:.3f}"
            end = '\n'
            if (self.counter.ep < self.counter.n_ep): end = '\r'
            print('# Ep #'+str(self.counter.ep)+', avg score = '+str(avg)+'      ', end=end)

    # Init temporary data
    def init_tmp_data(self):

        # These values are temporary storage for report struct
        self.p_loss  = 0.0
        self.entropy = 0.0
        self.p_gnorm = 0.0
        self.kl_div  = 0.0
        self.v_loss  = 0.0
        self.v_gnorm = 0.0

    # Training
    def train(self):

        # Save policy weights
        self.policy.save_weights()

        # Train policy and v_value
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

                self.train_policy (btc_obs, btc_adv, btc_act)
                self.train_v_value(btc_obs, btc_tgt, end - start)

        # Update old policy
        self.policy.set_prv_weights()

    ################################
    ### Policy/value wrappings
    ################################

    # Training function for policy
    def train_policy(self, obs, adv, act):

        outputs      = self.policy.train(obs, adv, act)
        self.p_loss  = outputs[0]
        self.kl_div  = outputs[1]
        self.p_gnorm = outputs[2]
        self.entropy = outputs[3]

    # Training function for critic
    def train_v_value(self, obs, tgt, size):

        outputs      = self.v_value.train(obs, tgt, size)
        self.v_loss  = outputs[0]
        self.v_gnorm = outputs[1]

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

        self.report.append("episode",       self.counter.ep)
        self.report.append("score",         self.counter.score[cpu])
        self.report.append("length",        self.counter.ep_step[cpu])
        self.report.append("policy_loss",   self.p_loss)
        self.report.append("v_value_loss",  self.v_loss)
        self.report.append("entropy",       self.entropy)
        self.report.append("policy_gnorm",  self.p_gnorm)
        self.report.append("v_value_gnorm", self.v_gnorm)
        self.report.append("kl_div",        self.kl_div)
        self.report.append("policy_lr",     self.policy.get_lr())
        self.report.append("v_value_lr",    self.v_value.get_lr())

        self.report.step(self.counter.ep_step[cpu])

    # Write learning data report
    def write_report(self, path, run):

        # Set filename with method name and run number
        filename = path+'/'+self.name+'_'+str(run)+'.dat'
        self.report.write(filename, self.report_fields)

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
