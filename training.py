# Generic imports
import os
import time
import collections
import numpy as np

# Custom imports
from ppo         import *
#from ppo_cma     import *

# Start training
def launch_training(env_name,
                    n_episodes, n_steps, render_every,
                    learn_rate, buff_size, batch_size, n_epochs,
                    clip, entropy, gamma, gae_lambda, update_alpha):


    # Declare environement and agent
    env     = gym.make(env_name)
    #env     = gym.wrappers.Monitor(env,
    #                               './vids/'+str(time.time())+'/',
    #                               video_callable=lambda episode_id: episode_id%10==0)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    agent = ppo(act_dim, obs_dim, n_episodes,
                learn_rate, buff_size, batch_size, n_epochs,
                clip, entropy, gamma, gae_lambda, update_alpha)

    # Initialize buffers
    buff_obs = np.zeros((buff_size, obs_dim))
    buff_act = np.zeros((buff_size, act_dim))
    buff_rwd = np.zeros((buff_size))
    buff_val = np.zeros((buff_size))
    buff_dlt = np.zeros((buff_size))
    buff_tgt = np.zeros((buff_size))
    buff_adv = np.zeros((buff_size))
    buff_trm = np.zeros((buff_size), dtype=bool)

    # Loop over episodes
    for ep in range(n_episodes):

        # Reset env
        ep_rwd   = 0.0
        ep_lgt   = 0
        obs      = env.reset()
        buff_cnt = 0

        # Loop over buff size
        for step in range(n_steps):

            # Make one iteration
            obs                   = np.clip(obs,-10,10)
            act, mu, sig          = agent.get_actions(obs)
            print(act)
            val                   = agent.get_value(obs)
            new_obs, rwd, done, _ = env.step(act)
            new_obs               = np.clip(new_obs,-10,10)
            new_val               = agent.get_value(new_obs)
            rwd                   = np.clip(rwd,-5,5)
            dlt                   = agent.compute_delta(rwd, val, new_val)

            # Store in buffers
            buff_obs[buff_cnt,:] = obs
            buff_act[buff_cnt,:] = act
            buff_rwd[buff_cnt]   = rwd
            buff_val[buff_cnt]   = val
            buff_dlt[buff_cnt]   = dlt
            buff_trm[buff_cnt]   = done

            # Update observation and buffer counter
            obs       = new_obs
            ep_rwd   += rwd
            ep_lgt   += 1
            buff_cnt += 1

            # Check if it is time for training
            if (buff_cnt == buff_size):

                # Compute deltas, targets and advantages
                agent.compute_targets   (buff_rwd, buff_tgt, buff_trm)
                agent.compute_advantages(buff_dlt, buff_adv, buff_trm)

                # Store buffers
                agent.store_buffers(buff_obs, buff_act, buff_rwd,
                                    buff_val, buff_dlt, buff_tgt,
                                    buff_adv)

                # Train networks
                agent.train_networks(buff_obs, buff_act, buff_adv, buff_tgt)

                # Reset buffers
                buff_cnt      = 0
                buff_obs[:,:] = 0.0
                buff_act[:,:] = 0.0
                buff_rwd[:]   = 0.0
                buff_val[:]   = 0.0
                buff_dlt[:]   = 0.0
                buff_tgt[:]   = 0.0
                buff_adv[:]   = 0.0
                buff_trm[:]   = False

            # Check if episode is over
            if (done):
                # Printings
                #if ((ep % render_every) == 0):
                 #   env.render()
                if (ep == n_episodes-1): end = '\n'
                if (ep != n_episodes-1): end = '\r'
                print('# Ep #'+str(ep)+', ep_rwd = '\
                    +str(ep_rwd)+', ep_lgt = '+str(ep_lgt))

                break
