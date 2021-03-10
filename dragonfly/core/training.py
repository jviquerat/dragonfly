# Generic imports
import time

# Custom imports
from dragonfly.agents.ppo    import *
from dragonfly.envs.par_envs import *

########################
# Process training
########################
def launch_training(params, path, run):

    # Declare environement and agent
    env   = par_envs(params.env_name, params.n_cpu, path)
    agent = ppo(env.act_dim, env.obs_dim, params)

    # Initialize parameters
    #ep      =  0
    #ep_step = [0     for _ in range(params.n_cpu)]
    #score   = [0.0   for _ in range(params.n_cpu)]
    obs     = env.reset()

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
            obs       = nxt
            agent.update_score(rwd)
            agent.update_step()
            #score[:] += rwd[:]
            #ep_step   = [x+1 for x in ep_step]

            # Handle rendering
            rnd = env.render(agent.get_render_cpu())
            agent.store_rendering(rnd)

            # Reset if episode is done
            for cpu in range(params.n_cpu):
                if done[cpu]:

                    # Store for future file printing
                    agent.store(cpu)

                    # Print
                    agent.print_episode()

                    # Handle rendering
                    agent.finish_rendering(path, cpu)

                    # Reset
                    obs[cpu] = env.reset_single(cpu)
                    agent.reset_ep(cpu)
                    #score[cpu]   = 0
                    #ep_step[cpu] = 0
                    #ep          += 1

        # Train networks
        agent.train_networks()

        # Write report data to file
        agent.write_report(path, run)

    # Last printing
    agent.print_episode()

    # Close environments
    env.close()
