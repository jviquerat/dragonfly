# Generic imports
import os
import numpy as np

# Custom imports
from ppo         import *
#from ppo_cma     import *

# Start training
def launch_training(actor, env_name,
                    n_episodes, n_steps, render_every,
                    learn_rate, buff_size, batch_size, n_epochs,
                    clip, entropy, gamma, gae_lambda, update_alpha):


    # Declare environement and agent
    env     = gym.make(env_name)
    act_dim = env.action_space.shape
    obs_dim = env.observation_space.shape

    if (actor == 'ppo'):
        agent = ppo(act_dim, obs_dim, n_episodes, n_steps,
                    learn_rate, buff_size, batch_size, n_epochs,
                    clip, entropy, gamma, gae_lambda, update_alpha)

    # Initialize buffers
    buff_obs = np.zeros(buff_size, obs_dim)
    buff_act = np.zeros(buff_size, act_dim)
    buff_rwd = np.zeros(buff_size)
    buff_dlt = np.zeros(buff_size)
    buff_val = np.zeros(buff_size)
    buff_tgt = np.zeros(buff_size)

    # Initialize parameters
    ep   = 0
    done = True

    # Loop over episodes
    while (ep < n_episodes):

        # Reset environment
        if done:
            buff_cnt      = 0
            done          = False
            obs           = env.reset()
            buff_obs[:,:] = 0.0
            buff_act[:,:] = 0.0
            buff_rwd[:]   = 0.0
            buff_dlt[:]   = 0.0
            buff_val[:]   = 0.0
            buff_tgt[:]   = 0.0

        # Loop while episode not over and buff not full
        while ((not done) and (buff_cnt < buff_size)):

            # Make one iteration
            act, mu, sig          = agent.get_actions(obs)
            val                   = agent.get_value(obs)
            new_obs, rwd, done, _ = env.step(act)
            new_val               = agent.get_value(new_obs)
            dlt                   = agent.compute_delta(rwd, val, new_val)

            # Store in buffers
            buff_obs[buff_cnt,:]  = obs
            buff_act[buff_cnt,:]  = act
            buff_rwd[buff_cnt]    = rwd
            buff_dlt[buff_cnt]    = dlt
            buff_val[buff_cnt]    = val

            #agent.store_transition(obs, act, rwd, val, dlt)

            # Update observation and buffer counter
            obs       = new_obs
            buff_cnt += 1

        # Handle value of last state
        if (    terminal): target_val = 0
        if (not terminal): target_val = agent.get_value(obs)

        # Compute target values using reversed reward buffer
        rev_rwd = buff_rwd.reverse()
        for i in range(buff_size):
            target_val  = rev_rwd[i] + gamma*target_val
            buff_tgt[i] = target_val
        buff_tgt.reverse()



            # Store a few things
            #agent.ep [episode] = episode
            #agent.gen[episode] = gen

            #if (rwd > bst_rwd):
            #    bst_rwd  = rwd
            #    bst_cact = cact

            # Update global index
            #episode += 1

        # Store a few things
        #agent.bst_gen [gen] = gen
        #agent.bst_ep  [gen] = episode
        #agent.bst_rwd [gen] = bst_rwd
        #agent.bst_cact[gen] = bst_cact

        # Train network after one generation
        agent.compute_advantages()
        agent.train_network()

        # Printings
        if (ep == n_episodes-1): end = '\n'
        if (ep != n_episodes-1): end = '\r'
        print('#   Episode #'+str(ep)+', rwd_sum = '+str(rwd_sum), end=end)

    # # Write to files
    # filename = 'database.opt.dat'
    # np.savetxt(filename, np.transpose([agent.gen,
    #                                    agent.ep,
    #                                    agent.cact[:,0],
    #                                    agent.cact[:,1],
    #                                    agent.rwd*(-1.0),
    #                                    np.zeros(n_gen*n_ind)]))

    # filename = 'optimisation.dat'
    # np.savetxt(filename, np.transpose([agent.bst_gen+1,
    #                                    agent.bst_ep,
    #                                    agent.bst_cact[:,0],
    #                                    agent.bst_cact[:,1],
    #                                    agent.bst_rwd*(-1.0)]))
