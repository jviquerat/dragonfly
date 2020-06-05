# Generic imports
import os
import time
import numpy as np

# Custom imports
from params import *
from ppo    import *

########################
# Process training
########################
def launch_training():

    # Declare environement and agent
    env     = gym.make(env_name)
    video   = lambda episode_id: episode_id%render_every==0
    env     = gym.wrappers.Monitor(env,
                                   './vids/'+str(time.time())+'/',
                                   video_callable=video)
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    agent   = ppo_discrete(act_dim, obs_dim, actor_lr, critic_lr,
                           buff_size, batch_size, n_epochs, n_buff,
                           pol_clip, grd_clip, adv_clip, bootstrap,
                           entropy, gamma, gae_lambda,
                           actor_arch, critic_arch)

    # Initialize parameters
    ep      = 0
    step    = 0
    score   = 0.0
    mask    = 1.0
    outputs = 8*[0.0]
    obs     = env.reset()

    # Loop until max episode number is reached
    while (ep < n_episodes):

        # Reset local buffers
        agent.reset_local_buffers()

        # Loop over buff size
        for _ in range(buff_size):

            # Make one iteration
            act               = agent.get_actions(obs)
            nxt, rwd, done, _ = env.step(np.argmax(act))
            step             += 1

            # Store transition
            if (not bootstrap): mask = float(not done)
            if (    bootstrap):
                if (not done):                    mask = 1.0
                if (    done and step <  ep_end): mask = 0.0
                if (    done and step == ep_end): mask = 1.0

            agent.store_transition(obs, nxt, act, rwd, mask)

            # Update observation and buffer counter
            obs       = nxt
            score    += rwd

            # Reset if episode is done
            if done:
                # Store for future file printing
                agent.store_learning_data(ep, score, outputs)

                # Print and reset
                avg = np.mean(agent.score[-25:])
                avg = f"{avg:.3f}"
                print('# Ep #'+str(ep)+', avg score = '+str(avg), end='\r')
                obs   = env.reset()
                score = 0
                step  = 0
                ep   += 1

        # Train
        outputs = agent.train()

    # Write learning data on file
    agent.write_learning_data()

    # Last printing
    print('# Ep #'+str(ep)+', avg score = '+str(avg), end='\n')

########################
# Average training over multiple runs
########################

# Storage arrays
n_data    = 9
ep        = np.zeros((       n_episodes),           dtype=int)
data      = np.zeros((n_avg, n_episodes,   n_data), dtype=float)
avg_data  = np.zeros((       n_episodes,   n_data), dtype=float)
stdp_data = np.zeros((       n_episodes,   n_data), dtype=float)
stdm_data = np.zeros((       n_episodes,   n_data), dtype=float)

for i in range(n_avg):
    print('### Avg run #'+str(i))
    launch_training()

    f           = np.loadtxt('ppo.dat')
    ep          = f[:n_episodes,0]
    for j in range(n_data):
        data[i,:,j] = f[:n_episodes,j+1]

# Write to file
file_out  = 'ppo_avg.dat'
array     = np.vstack(ep)
for j in range(n_data):
    avg   = np.mean(data[:,:,j], axis=0)
    std   = np.std (data[:,:,j], axis=0)
    p     = avg + std
    m     = avg - std
    array = np.hstack((array,np.vstack(avg)))
    array = np.hstack((array,np.vstack(p)))
    array = np.hstack((array,np.vstack(m)))

np.savetxt(file_out, array)
os.system('gnuplot -c plot.gnu')
