# Generic imports
import os
import time
import collections
import numpy as np

# Custom imports
from params       import *
from ppo_discrete import *

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
                           buff_size, batch_size, n_epochs, l2_reg,
                           orth_gain, clip, entropy, gamma, gae_lambda,
                           alpha, actor_arch, critic_arch)

    # Initialize buffer-related parameters
    buff_obs = []
    buff_act = []
    buff_rwd = []
    buff_val = []
    buff_msk = []
    buff_cnt = 0

    # Loop over episodes
    for ep in range(n_episodes):

        # Reset episode-related parameters
        ep_rwd   = 0.0
        ep_lgt   = 0
        obs      = env.reset()
        done     = False

        # Loop over buff size
        while (not done):

            # Make one iteration
            act                   = agent.get_actions(obs)
            val                   = agent.get_value(obs)
            new_obs, rwd, done, _ = env.step(np.argmax(act))

            # Store in local buffers
            buff_obs.append(obs)
            buff_act.append(act)
            buff_rwd.append(rwd)
            buff_val.append(val)
            buff_msk.append(float(not done))

            # Update observation and buffer counter
            obs       = new_obs
            ep_rwd   += rwd
            ep_lgt   += 1
            buff_cnt += 1

            # Check if it is time for training
            #if (buff_cnt == buff_size):
            if done:

                # Buffers are full, proceed to training
                buff_act = np.vstack(buff_act)
                buff_rwd = np.array(buff_rwd)
                buff_obs = np.vstack(buff_obs)
                buff_val = np.array(buff_val)

                buff_tgt = agent.compute_tgts(buff_rwd, buff_msk)
                buff_adv = agent.compute_advs(buff_rwd, buff_val, buff_msk)

                agent.train_networks(buff_obs, buff_act,
                                     buff_adv, buff_tgt)


                # Reset buffers
                buff_obs = []
                buff_act = []
                buff_rwd = []
                buff_val = []
                buff_msk = []
                buff_cnt = 0

        # Store in global buffers
        agent.eps.append(ep)
        agent.rwd.append(ep_rwd)
        agent.lgt.append(ep_lgt)
        #agent.val.extend(buff_val)
        #agent.tgt.extend(buff_tgt)
        #agent.adv.extend(buff_adv)

        # Printings
        print('# Ep #'+str(ep)+', ep_rwd = '+str(ep_rwd)+', ep_lgt = '+str(ep_lgt))

    # Write global buffers
    filename = 'ppo.dat'
    #print(agent.stp)
    #print(agent.rwd)
    #print(agent.val)
    #print(agent.tgt)
    #print(agent.adv)
    np.savetxt(filename, np.transpose([np.asarray(agent.eps),
                                       np.asarray(agent.rwd),
                                       np.asarray(agent.lgt)]))


# Average results over multiple runs
# idx     = np.zeros((      n_gen), dtype=int)
# cost    = np.zeros((n_avg,n_gen), dtype=float)

for i in range(n_avg):
    print('### Avg run #'+str(i))
    launch_training()

    #f         = np.loadtxt('optimisation.dat')
    #idx       = f[:,0]
    #cost[i,:] = f[:,4]

# Write to file
# file_out = 'ppo_avg_data.dat'
# avg      = np.mean(cost,axis=0)
# std      = 0.5*np.std (cost,axis=0)

# # Be careful about standard deviation plotted in log scale
# log_avg  = np.log(avg)
# log_std  = 0.434*std/avg
# log_p    = log_avg+log_std
# log_m    = log_avg-log_std
# p        = np.exp(log_p)
# m        = np.exp(log_m)

# array    = np.transpose(np.stack((idx, avg, m, p)))
# np.savetxt(file_out, array)
