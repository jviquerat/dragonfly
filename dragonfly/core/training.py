# Generic imports
import time
import numpy as np

########################
# Process training
########################
def launch_training(params, path, run, env, agent):

    # Reset environment
    obs = env.reset_all()

    # Loop until max episode number is reached
    while (agent.test_ep_loop()):

        # Reset buffer
        agent.reset_buff()

        # Loop over buff size
        while (agent.test_buff_loop()):

            # Make one iteration
            act            = agent.get_actions(obs)
            nxt, rwd, done = env.step(np.argmax(act, axis=1))

            # Handle termination state
            trm, done      = agent.handle_term(done, params.ep_end)

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
            env.reset(done, obs)

        # Prepare training and train agent
        agent.finalize_buffers()
        agent.train()

        # Write report data to file
        agent.write_report(path, run)

    # Last printing
    agent.print_episode()
