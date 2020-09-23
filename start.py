# Generic imports
import os
import sys
import json
import time
import collections
import numpy as np

# Custom imports
from training import *

########################
# Parameters decoder to collect json file
########################
def params_decoder(p_dict):
    return collections.namedtuple('X', p_dict.keys())(*p_dict.values())

########################
# Average training over multiple runs
########################

# Check command-line input for json file
if (len(sys.argv) == 2):
    json_file = sys.argv[1]
else:
    print('Command line error, please use as follows:')
    print('python3 start.py my_file.json')

# Read json parameter file
with open(json_file, "r") as f:
    params = json.load(f, object_hook=params_decoder)

# Storage arrays
n_data     = 9
ep         = np.zeros((              params.n_ep),           dtype=int)
data       = np.zeros((params.n_avg, params.n_ep,   n_data), dtype=float)
avg_data   = np.zeros((              params.n_ep,   n_data), dtype=float)
stdp_data  = np.zeros((              params.n_ep,   n_data), dtype=float)
stdm_data  = np.zeros((              params.n_ep,   n_data), dtype=float)

for i in range(params.n_avg):
    print('### Avg run #'+str(i))
    start_time = time.time()
    launch_training(params)
    print("--- %s seconds ---" % (time.time() - start_time))

    f           = np.loadtxt('ppo.dat')
    ep          = f[:params.n_ep,0]
    for j in range(n_data):
        data[i,:,j] = f[:params.n_ep,j+1]

# Write to file
file_out  = 'ppo_avg.dat'
path = '.'
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
os.system('gnuplot -c plot/plot.gnu '+path)
