# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.utils.timer import *

########################
# Process training
########################
def launch_training(path, run, env, agent):

    # Initialize timers
    timer_global   = timer("global   ")
    timer_env      = timer("env      ")
    timer_actions  = timer("actions  ")
    timer_training = timer("training ")

    # Start global timer
    timer_global.tic()

    # Reset environment
    obs = env.reset_all()

    # Loop until max episode number is reached
    while (agent.test_ep_loop()):

        # Reset buffer
        agent.reset_buff()

        # Loop over buff size
        while (agent.test_buff_loop()):

            # Get actions
            timer_actions.tic()
            act = agent.get_actions(obs)
            timer_actions.toc()

            # Make one env step
            timer_env.tic()
            nxt, rwd, done = env.step(act)
            timer_env.toc()

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
            timer_env.tic()
            env.reset(done, obs)
            timer_env.toc()

        # Finalize buffers for training
        agent.finalize_buffers()

        # Train agent
        timer_training.tic()
        agent.train()
        timer_training.toc()

        # Write report data to file
        agent.write_report(path, run)

    # Last printing
    agent.print_episode()

    # Close timers and show
    timer_global.toc()
    timer_global.show()
    timer_env.show()
    timer_actions.show()
    timer_training.show()
