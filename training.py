# Generic imports
import time

# Custom imports
from ppo import *

########################
# Process training
########################
def launch_training(params):

    # Declare environement and agent
    env     = gym.make(params.env_name)
    video   = lambda ep: (ep%params.render_every==0 and ep != 0)
    env     = gym.wrappers.Monitor(env,
                                   './vids/'+str(time.time())+'/',
                                   video_callable=video)
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    agent   = ppo_discrete(act_dim, obs_dim, params)

    # Initialize parameters
    ep      = 0
    ep_step = 0
    bf_step = 0
    score   = 0.0
    mask    = 1.0
    outputs = 8*[0.0]
    obs     = env.reset()

    # Loop until max episode number is reached
    while (ep < params.n_ep):

        # Reset local buffers
        agent.reset_local_buffers()
        bf_step = 0
        loop    = True

        # Loop over buff size
        while (loop):

            # Make one iteration
            act               = agent.get_actions(obs)
            nxt, rwd, done, _ = env.step(np.argmax(act))

            # Handle termination state
            if (not params.bootstrap):
                if (not done): term = 0
                if (    done): term = 1
            if (    params.bootstrap):
                if (not done):                                term = 0
                if (    done and ep_step <  params.ep_end-1): term = 1
                if (    done and ep_step == params.ep_end-1): term = 2

            # Store transition
            agent.store_transition(obs, nxt, act, rwd, term)

            # Update observation and buffer counter
            obs       = nxt
            score    += rwd

            # Reset if episode is done
            if done:
                # Store for future file printing
                agent.store_learning_data(ep, ep_step+1, score, outputs)

                # Print and reset
                avg     = np.mean(agent.score[-25:])
                avg     = f"{avg:.3f}"
                print('# Ep #'+str(ep)+', avg score = '+str(avg), end='\r')
                obs     = env.reset()
                score   = 0
                ep_step = 0
                ep     += 1
            else:
                ep_step +=1

            # Test if loop is over
            loop     = agent.test_loop(done, bf_step)
            bf_step += 1

        # Train
        outputs = agent.train()

    # Write learning data on file
    agent.write_learning_data()

    # Last printing
    print('# Ep #'+str(ep)+', avg score = '+str(avg), end='\n')
