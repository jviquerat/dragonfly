# Generic imports
import time

# Custom imports
from ppo      import *
from buff     import *
from par_envs import *

########################
# Process training
########################
def launch_training(params):

    # Declare environement and agent
    env = par_envs(params.env_name, params.n_cpu)
    #env     = gym.make(params.env_name)
    #video   = lambda ep: (ep%params.render_every==0 and ep != 0)
    #env     = gym.wrappers.Monitor(env,
    #                               './vids/'+str(time.time())+'/',
    #                               video_callable=video)
    buff    = loc_buff(params.n_cpu, env.obs_dim, env.act_dim)
    agent   = ppo_agent(env.act_dim, env.obs_dim, params)

    # Initialize parameters
    ep      = 0
    #ep_step = 0
    #score   = 0.0
    ep_step = [0   for _ in range(params.n_cpu)]
    score   = [0.0 for _ in range(params.n_cpu)]
    outputs = [0.0 for _ in range(8)]
    obs     = env.reset()

    # Loop until max episode number is reached
    while (ep < params.n_ep):

        # Reset local buffer
        #buff.reset()
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


            #act               = agent.get_actions(obs)
            #nxt, rwd, done, _ = env.step(np.argmax(act))

            # Handle termination state
            trm = np.array([])
            for cpu in range(params.n_cpu):
                out = agent.handle_termination(done[cpu],
                                               ep_step[cpu],
                                               params.ep_end)
                trm = np.append(trm,out)
            #trm = agent.handle_termination(done, ep_step, params.ep_end)

            # Store transition
            agent.loc_buff.store(obs, nxt, act, rwd, trm)
            #buff.store(obs, nxt, act, rwd, trm)

            # Update observation and buffer counter
            obs       = nxt
            score[:]  += rwd[:]
            #score    += rwd
            ep_step = [x+1 for x in ep_step]
            #ep_step  += 1

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

                    # Reset
                    obs[cpu]     = env.reset_single(cpu)
                    score[cpu]   = 0
                    ep_step[cpu] = 0
                    ep          += 1
            # if done:
            #     # Store for future file printing
            #     agent.store_learning_data(ep, ep_step, score, outputs)

            #     # Print
            #     agent.print_episode(ep, params.n_ep)

            #     # Reset
            #     obs     = env.reset()
            #     score   = 0
            #     ep_step = 0
            #     ep     += 1

            # Test if loop is over
            loop = agent.test_loop()

        # Train
        #buff.reshape()
        outputs = agent.train()

        # Write learning data on file
        agent.write_learning_data()

        # Last printing
        agent.print_episode(ep, params.n_ep)
