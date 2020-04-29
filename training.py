# Generic imports
import os
import time
import collections
import numpy as np

# Custom imports
from ppo_discrete import *

# Start training
def launch_training(env_name, alg_type,
                    n_episodes, n_steps, render_every,
                    actor_lr, critic_lr, buff_size, batch_size, n_epochs,
                    clip, entropy, gamma, gae_lambda, alpha):


    # Declare environement and agent
    env     = gym.make(env_name)
    video   = lambda episode_id: episode_id%render_every==0
    env     = gym.wrappers.Monitor(env,
                                   './vids/'+str(time.time())+'/',
                                   video_callable=video)
    print(env.action_space.shape)
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    agent = ppo_discrete(act_dim, obs_dim, n_episodes,
                         actor_lr, critic_lr, buff_size, batch_size, n_epochs,
                         clip, entropy, gamma, gae_lambda, alpha)

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

        # Loop over buff size
        while (not done):

            # Make one iteration
            act                   = agent.get_actions(obs)
            val                   = agent.get_value(obs)
            new_obs, rwd, done, _ = env.step(np.argmax(act))

            # Store in buffers
            buff_obs.append(obs)
            buff_act.append(act)
            buff_rwd.append(rwd)
            buff_val.append(val)

            # Update observation and buffer counter
            obs       = new_obs
            ep_rwd   += rwd
            ep_lgt   += 1

        # Episode is finished, proceed to training
        buff_act = np.vstack(buff_act)
        buff_rwd = np.array(buff_rwd)
        buff_obs = np.vstack(buff_obs)
        buff_val = np.array(buff_val)

        buff_tgt = agent.compute_tgts(buff_rwd)
        buff_adv = agent.compute_advs(buff_rwd, buff_val)

        agent.train_networks(buff_obs, buff_act,
                             buff_adv, buff_tgt)

        # Printings
        print('# Ep #'+str(ep)+', ep_rwd = '+str(ep_rwd)+', ep_lgt = '+str(ep_lgt))
