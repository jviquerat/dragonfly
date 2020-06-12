# Generic imports
import time
import multiprocessing as mp
from   multiprocessing.managers import BaseManager

# Custom imports
from ppo  import *
from buff import *

########################
# Process training
########################
def launch_training(params):

    # Declare environement and agent
    env     = gym.make(params.env_name)
    #video   = lambda ep: (ep%params.render_every==0 and ep != 0)
    #env     = gym.wrappers.Monitor(env,
    #                               './vids/'+str(time.time())+'/',
    #                               video_callable=video)
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    buff    = loc_buff(params.n_cpu, obs_dim, act_dim)
    #agent   = ppo_agent(act_dim, obs_dim, params)

    # Generate manager
    class agentManager(BaseManager):
        pass

    agentManager.register('ppo_agent', ppo_agent)

    if __name__ == '__main__':
        manager = agentManager()
        manager.start()
        agent = manager.ppo_agent(act_dim, obs_dim, params)

        pool = mp.Pool(2)
        for i in range(33):
            pool.apply(func=worker, args=(shared, i))
        pool.close()
        pool.join()

        exit()

        # Initialize parameters
        ep      = 0
        ep_step = 0
        score   = 0.0
        outputs = [0.0 for _ in range(8)]
        obs     = env.reset()

        # Loop until max episode number is reached
        while (ep < params.n_ep):

            # Reset local buffer
            buff.reset()
            loop = True

            # Loop over buff size
            while (loop):

                # Make one iteration
                act               = agent.get_actions(obs)
                nxt, rwd, done, _ = env.step(np.argmax(act))

                # Handle termination state
                trm = agent.handle_termination(done, ep_step, params.ep_end)

                # Store transition
                buff.store(obs, nxt, act, rwd, trm)

                # Update observation and buffer counter
                obs       = nxt
                score    += rwd
                ep_step  += 1

                # Reset if episode is done
                if done:
                    # Store for future file printing
                    agent.store_learning_data(ep, ep_step, score, outputs)

                    # Print
                    agent.print_episode(ep, params.n_ep)

                    # Reset
                    obs     = env.reset()
                    score   = 0
                    ep_step = 0
                    ep     += 1

                # Test if loop is over
                loop = agent.test_loop(done, buff.size)

                # Train
                buff.reshape()
                outputs = agent.train(buff)

            # Write learning data on file
            agent.write_learning_data()

            # Last printing
            agent.print_episode(ep, params.n_ep)
