# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.utils.timer import *

###############################################
### Class for buffer-based training
### pms : parameters
def buffer_based(pms):
    def __init__(self, pms):

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_env      = timer("env      ")
        self.timer_actions  = timer("actions  ")
        self.timer_training = timer("training ")

    # Train
    def train(self, path, run, env, agent):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = env.reset_all()

        # Loop until max episode number is reached
        while (agent.test_ep_loop()):

            # Reset buffer
            agent.reset_buff()

            # Loop over buff size
            while (agent.test_buff_loop()):

                # Get actions
                self.timer_actions.tic()
                act = agent.get_actions(obs)
                self.timer_actions.toc()

                # Make one env step
                self.timer_env.tic()
                nxt, rwd, done = env.step(act)
                self.timer_env.toc()

                # Handle termination state
                trm, done = agent.handle_term(done)

                # Store transition
                agent.store_transition(obs, nxt, act, rwd, trm)

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
            agent.finalize_buffers()

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
