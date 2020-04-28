# Generic imports
import os
import time
import collections
import numpy as np

# Custom imports
from ppo         import *
#from ppo_cma     import *

# Start training
def launch_training(env_name, alg_type,
                    n_episodes, n_steps, render_every,
                    learn_rate, buff_size, batch_size, n_epochs,
                    clip, entropy, gamma, gae_lambda, update_alpha):


    # Declare environement and agent
    env     = gym.make(env_name)
    env     = gym.wrappers.Monitor(env,
                                   './vids/'+str(time.time())+'/',
                                   video_callable=lambda episode_id: episode_id%50==0)
    print(env.action_space.shape)
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    agent = ppo(alg_type, act_dim, obs_dim, n_episodes,
                learn_rate, buff_size, batch_size, n_epochs,
                clip, entropy, gamma, gae_lambda, update_alpha)

    # Initialize buffer

    # Loop over episodes
    for ep in range(n_episodes):

        # Reset env
        ep_rwd   = 0.0
        ep_lgt   = 0
        obs      = env.reset()
        done     = False

        buff_obs = []
        buff_act = []
        buff_rwd = []
        buff_val = []
        #buff_dlt = []
        #buff_tgt = []
        #buff_adv = []

        # Loop over buff size
        while (not done):

            # Make one iteration
            #obs                   = np.clip(obs,-10,10)
            act                   = agent.get_actions(obs)
            val                   = agent.get_value(obs)
            new_obs, rwd, done, _ = env.step(np.argmax(act))
            #new_obs               = np.clip(new_obs,-10,10)
            #new_val               = agent.get_value(new_obs)
            #rwd                   = np.clip(rwd,-5,5)

            # Store in buffers
            buff_obs.append(obs)
            buff_act.append(act)
            buff_rwd.append(rwd)
            buff_val.append(val)

            # Update observation and buffer counter
            obs       = new_obs
            ep_rwd   += rwd
            ep_lgt   += 1

            # Check if it is time for training
            #if done:

        buff_act = np.vstack(buff_act)
        buff_rwd = np.array(buff_rwd)
        buff_obs = np.vstack(buff_obs)
        buff_val = np.array(buff_val)

        buff_tgt = agent.compute_tgts(buff_rwd)
        buff_adv = agent.compute_advs(buff_rwd, buff_val)

                # Compute deltas, targets and advantages
                #agent.compute_dlt_tgt_adv(buff_rwd, buff_tgt, buff_dlt,
                #                          buff_adv, buff_val, buff_msk)

                # Store buffers
                #agent.store_buffers(buff_obs, buff_act, buff_rwd,
                #                    buff_val, buff_dlt, buff_tgt,
                #                    buff_adv)

        # Train networks
        agent.train_networks(buff_obs, buff_act, buff_adv, buff_tgt)

        # Reset buffers
        buff_obs = []
        buff_act = []
        buff_rwd = []
        buff_val = []

            # Check if episode is over
            #if (done or (step == n_steps-1)):
                # Printings
                #if ((ep % render_every) == 0):
                 #   env.render()
        if (ep == n_episodes-1): end = '\n'
        if (ep != n_episodes-1): end = '\r'
        print('# Ep #'+str(ep)+', ep_rwd = '+str(ep_rwd)+', ep_lgt = '+str(ep_lgt))
