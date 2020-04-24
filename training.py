# Generic imports
import os
import numpy as np

# Custom imports
from environment import *
from ppo         import *
from ppo_cma     import *

# Start training
def launch_training(actor, n_params, init_obs,
                    x_min, x_max, y_min, y_max,
                    n_gen, n_ind,
                    clip, entropy, learn_rate, actor_epochs,
                    n_batch, clip_adv, mirror_adv):

    # Declare environement and agent
    env   = env_opt(init_obs, n_params, x_min, x_max, y_min, y_max)
    if (actor == 'ppo'):     agent = ppo    (n_params, n_params,
                                             n_gen, n_ind,
                                             clip, entropy,
                                             learn_rate, actor_epochs)
    if (actor == 'ppo-cma'): agent = ppo_cma(n_params, n_params,
                                             n_gen, n_ind, n_batch,
                                             learn_rate, actor_epochs,
                                             clip_adv, mirror_adv)

    # Initialize parameters
    episode   = 0
    bst_cact = np.zeros(n_params)
    bst_rwd  = -1.0e10

    # Loop over generations
    for gen in range(n_gen):

        # Printings
        if (gen == n_gen-1): end = '\n'
        if (gen != n_gen-1): end = '\r'
        print('#   Generation #'+str(gen), end=end)

        # Loop over individuals
        for ind in range(n_ind):

            # Make one iteration
            obs          = env.reset()
            act, mu, sig = agent.get_actions(obs)
            rwd, cact    = env.step(act)
            agent.store_transition(obs, act, cact, rwd, mu, sig)

            # Store a few things
            agent.ep [episode] = episode
            agent.gen[episode] = gen

            if (rwd > bst_rwd):
                bst_rwd  = rwd
                bst_cact = cact

            # Update global index
            episode += 1

        # Store a few things
        agent.bst_gen [gen] = gen
        agent.bst_ep  [gen] = episode
        agent.bst_rwd [gen] = bst_rwd
        agent.bst_cact[gen] = bst_cact

        # Train network after one generation
        agent.compute_advantages()
        agent.train_network()

    # Write to files
    filename = 'database.opt.dat'
    np.savetxt(filename, np.transpose([agent.gen,
                                       agent.ep,
                                       agent.cact[:,0],
                                       agent.cact[:,1],
                                       agent.rwd*(-1.0),
                                       np.zeros(n_gen*n_ind)]))

    filename = 'optimisation.dat'
    np.savetxt(filename, np.transpose([agent.bst_gen+1,
                                       agent.bst_ep,
                                       agent.bst_cact[:,0],
                                       agent.bst_cact[:,1],
                                       agent.bst_rwd*(-1.0)]))
