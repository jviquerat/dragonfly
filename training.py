# Generic imports
import time
from   PIL import Image

# Custom imports
from ppo      import *
from buff     import *
from par_envs import *

########################
# Process training
########################
def launch_training(params):

    # Declare environement and agent
    env   = par_envs(params.env_name, params.n_cpu)
    agent = ppo_agent(env.act_dim, env.obs_dim, params)

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

        # Reset local buffer
        agent.loc_buff.reset()
        loop = True

        # Loop over buff size
        while (loop):

            # Make one iteration
            act = np.array([])
            for cpu in range(params.n_cpu):
                out = agent.get_actions(obs[cpu])
                act = np.append(act,out)
            act = np.reshape(act, (-1,agent.act_dim))
            nxt, rwd, done = env.step(np.argmax(act, axis=1))

            # Handle termination state
            trm = np.array([])
            for cpu in range(params.n_cpu):
                term, done[cpu] = agent.handle_termination(done[cpu],
                                                           ep_step[cpu],
                                                           params.ep_end)
                trm = np.append(trm, term)

            # Store transition
            agent.loc_buff.store(obs, nxt, act, rwd, trm)

            # Update observation and buffer counter
            obs       = nxt
            score[:] += rwd[:]
            ep_step   = [x+1 for x in ep_step]

            # Handle rendering
            for cpu in range(params.n_cpu):
                if (render[cpu]):
                    rgb[cpu].append(Image.fromarray(env.render_single(cpu)))

            # Reset if episode is done
            for cpu in range(params.n_cpu):
                if done[cpu]:

                    # Store for future file printing
                    agent.store_learning_data(ep,
                                              ep_step[cpu],
                                              score[cpu],
                                              outputs)

                    # Print
                    agent.print_episode(ep, params.n_ep)

                    # Handle rendering
                    if (render[cpu]):
                        render[cpu] = False
                        rgb[cpu][0].save('vids/'+str(ep)+'.gif',
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

        # Train
        outputs = agent.train()

        # Write learning data on file
        agent.write_learning_data()

    # Last printing
    agent.print_episode(ep, params.n_ep)

    # Close environments
    env.close()
