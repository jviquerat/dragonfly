# Generic imports
import numpy as np

# Custom imports
from params   import *
from training import *

# Average results over multiple runs
# idx     = np.zeros((      n_gen), dtype=int)
# cost    = np.zeros((n_avg,n_gen), dtype=float)

for i in range(n_avg):
    print('### Avg run #'+str(i))
    launch_training(actor, env_name,
                    n_episodes, max_ep_steps, render_every,
                    learn_rate, batch_size, actor_epochs,
                    clip, entropy, gamma, gae_lambda, update_alpha)

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
