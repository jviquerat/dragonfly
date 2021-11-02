# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.utils.timer import *
from dragonfly.src.utils.buff  import *

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

        # pol_act_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.loc_buff = loc_buff(self.n_cpu,       self.obs_dim,
                                 self.pol_act_dim, self.buff_size)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_env      = timer("env      ")
        self.timer_actions  = timer("actions  ")
        self.timer_training = timer("training ")

    # Reset
    def reset(self):

        pass

    # Reset
    def reset_loc_buff(self):

        self.loc_buff.reset()
        #pass

    # Test buffer loop criterion
    def test_buff_loop(self):

        return self.loc_buff.test_buff_loop()

    # Store transition in local buffer
    def store_transition(self, obs, nxt, act, rwd, trm, bts):

        self.loc_buff.store(obs, nxt, act, rwd, trm, bts)

    # Train
    def train(self, path, run, env, agent):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = env.reset_all()

        # Loop until max episode number is reached
        while (agent.test_ep_loop()):

            # Reset local buffer
            self.reset_loc_buff()

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
                agent.finish_episodes(path, done)

                # Reset only finished environments
                self.timer_env.tic()
                env.reset(done, obs)
                self.timer_env.toc()

            # Finalize buffers for training
            #agent.finalize_buffers()
            self.loc_buff.fix_trm_buffer()
            obs, nxt, act, rwd, trm, bts = self.loc_buff.serialize()
            agent.compute_returns(obs, nxt, act, rwd, trm, bts)

            # Train agent
            self.timer_training.tic()
            agent.train()
            self.timer_training.toc()

            # Write report data to file
            agent.write_report(path, run)

        # Last printing
        agent.print_episode()

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.timer_env.show()
        self.timer_actions.show()
        self.timer_training.show()
