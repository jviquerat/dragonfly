# Generic imports
import time

# Custom imports
from dragonfly.agents.ppo    import *
from dragonfly.envs.par_envs import *
#from dragonfly.core.renderer import *

########################
# Process training
########################
def launch_training(params, path, run):

    # Declare environement and agent
    env   = par_envs(params.env_name, params.n_cpu, path)
    agent = ppo(env.act_dim, env.obs_dim, params)

    # Declare renderer
    #rnd = renderer(params.n_cpu)

    # Initialize parameters
    ep      =  0
    ep_step = [0     for _ in range(params.n_cpu)]
    score   = [0.0   for _ in range(params.n_cpu)]
    render  = [False for _ in range(params.n_cpu)]
    rgb     = [[]    for _ in range(params.n_cpu)]
    outputs = [0.0   for _ in range(8)]
    obs     = env.reset()

    # Loop until max episode number is reached
    while (ep < params.n_ep):

        # Reset buffer
        agent.reset_buff()
        loop = True

        # Loop over buff size
        while (loop):

            # Make one iteration
            act            = agent.get_actions(obs)
            nxt, rwd, done = env.step(np.argmax(act, axis=1))

            # Handle termination state
            trm, done      = agent.handle_term(done, ep_step, params.ep_end)

            # Store transition
            agent.store_transition(obs, nxt, act, rwd, trm)

            # Update observation and buffer counter
            obs       = nxt
            score[:] += rwd[:]
            ep_step   = [x+1 for x in ep_step]

            # Handle rendering
            rgb = env.render(render, rgb)

            # Reset if episode is done
            for cpu in range(params.n_cpu):
                if done[cpu]:

                    # Store for future file printing
                    agent.store(episode      = ep,
                                score        = score[cpu],
                                length       = ep_step[cpu])
                                # actor_loss   = outputs[0],
                                # critic_loss  = outputs[4],
                                # entropy      = outputs[1],
                                # actor_gnorm  = outputs[2],
                                # critic_gnorm = outputs[5],
                                # kl_div       = outputs[3],
                                # actor_lr     = outputs[6],
                                # critic_lr    = outputs[7])

                    # Print
                    agent.print_episode(ep, params.n_ep)

                    # Handle rendering
                    if (render[cpu]):
                        render[cpu] = False
                        rgb[cpu][0].save(path+'/'+str(ep)+'.gif',
                                         save_all=True,
                                         append_images=rgb[cpu][1:],
                                         optimize=False,
                                         duration=50,
                                         loop=1)
                        rgb[cpu] = []

                    if ((ep%params.render_every == 0) and (ep != 0)):
                        render[cpu] = True

                    # Reset
                    obs[cpu]     = env.reset_single(cpu)
                    score[cpu]   = 0
                    ep_step[cpu] = 0
                    ep          += 1

            # Test if loop is over
            loop = agent.test_loop()

        # Train networks
        agent.train_networks()

        # Write report data to file
        agent.write_report(path, run)

    # Last printing
    agent.print_episode(ep, params.n_ep)

    # Close environments
    env.close()
