# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.utils.timer    import *
from dragonfly.src.utils.buff     import *
from dragonfly.src.core.constants import *


###############################################
### Class for buffer-based training
### pms : parameters
class buffer_based():
    def __init__(self, obs_dim, act_dim, pol_act_dim, pms):

        # Initialize from input
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.pol_act_dim = pol_act_dim
        self.n_cpu       = pms.n_cpu
        self.buff_size   = pms.buff_size
        self.n_buff      = pms.n_buff
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs

        # pol_act_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.loc_buff = loc_buff(self.n_cpu,       self.obs_dim,
                                 self.pol_act_dim, self.buff_size)
        self.glb_buff = glb_buff(self.n_cpu,       self.obs_dim,
                                 self.pol_act_dim, self.n_buff,
                                 self.buff_size,   self.btc_frac)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_env      = timer("env      ")
        self.timer_actions  = timer("actions  ")
        self.timer_training = timer("training ")

    # Loop
    def loop(self, path, run, env, agent):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = env.reset_all()

        # Loop until max episode number is reached
        while (agent.test_ep_loop()):

            # Reset local buffer
            self.loc_buff.reset()

            # Loop over buff size
            while (self.test_buff_loop()):

                # Get actions
                self.timer_actions.tic()
                act = agent.get_actions(obs)
                self.timer_actions.toc()

                # Make one env step
                self.timer_env.tic()
                nxt, rwd, done = env.step(act)
                self.timer_env.toc()

                # Handle termination state
                trm, bts = agent.handle_term(done)

                # Store transition
                self.store_transition(obs, nxt, act, rwd, trm, bts)

                # Update observation and buffer counter
                obs = nxt
                agent.update_score(rwd)
                agent.update_step()

                # Handle rendering
                rnd = env.render(agent.get_render_cpu())
                agent.store_rendering(rnd)

                # Finish if some episodes are done
                self.finish_episodes(agent, path, done)

                # Reset only finished environments
                self.timer_env.tic()
                env.reset(done, obs)
                self.timer_env.toc()

            # Finalize buffers for training
            self.loc_buff.fix_trm_buffer()
            obs, nxt, act, rwd, trm, bts = self.loc_buff.serialize()
            tgt, adv = agent.compute_returns(obs, nxt, act, rwd, trm, bts)

            # Store in global buffers
            self.glb_buff.store(obs, adv, tgt, act)

            # Train agent
            self.timer_training.tic()
            self.train(agent)
            self.timer_training.toc()

            # Write report data to file
            agent.write_report(path, run)

        # Last printing
        self.print_episode(agent.counter, agent.report)

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.timer_env.show()
        self.timer_actions.show()
        self.timer_training.show()

    # Train
    def train(self, agent):

        # Save previous policy
        agent.policy.save_prv()

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

                agent.train(btc_obs, btc_act, btc_adv, btc_tgt, end-start)

    # Reset
    def reset(self):

        self.loc_buff.reset()
        self.glb_buff.reset()

    # Test buffer loop criterion
    def test_buff_loop(self):

        return self.loc_buff.test_buff_loop()

    # Store transition in local buffer
    def store_transition(self, obs, nxt, act, rwd, trm, bts):

        self.loc_buff.store(obs, nxt, act, rwd, trm, bts)

    # Finish if some episodes are done
    def finish_episodes(self, agent, path, done):

        # Loop over environments and finalize/reset
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                agent.store_report(cpu)
                self.print_episode(agent.counter, agent.report)
                agent.finish_rendering(path, cpu)
                agent.counter.reset_ep(cpu)

    # Printings at the end of an episode
    def print_episode(self, counter, report):

        # No initial printing
        if (counter.ep == 0): return

        # Average and print
        if (counter.ep <= counter.n_ep):
            avg    = np.mean(report.data["score"][-n_smooth:])
            avg    = f"{avg:.3f}"
            bst    = counter.best_score
            bst    = f"{bst:.3f}"
            bst_ep = counter.best_ep
            end    = '\n'
            if (counter.ep < counter.n_ep): end = '\r'
            print('# Ep #'+str(counter.ep)+', avg score = '+str(avg)+', best score = '+str(bst)+' at ep '+str(bst_ep)+'                 ', end=end)
