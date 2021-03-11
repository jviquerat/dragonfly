# Generic imports
import os
import sys
import time
import numpy as np

# Custom imports
from dragonfly.core.training import *
from dragonfly.core.utils    import *

########################
# Average training over multiple runs
########################
if __name__ == '__main__':

    # Check command-line input for json file
    if (len(sys.argv) == 2):
        json_file = sys.argv[1]
    else:
        print('Command line error, please use as follows:')
        print('python3 start.py my_file.json')

    # Initialize json parser and read parameters
    parser = json_parser()
    pms    = parser.read(json_file)

    # Create paths for results and open repositories
    res_path = 'results'
    t         = time.localtime()
    path_time = time.strftime("%H-%M-%S", t)
    path      = res_path+'/'+pms.env_name+'_'+str(path_time)

    # Open repositories
    if (not os.path.exists(res_path)):
        os.makedirs(res_path)
    if (not os.path.exists(path)):
        os.makedirs(path)

    # Intialize averager
    n_fields = 9
    averager = data_avg(n_fields, pms.n_ep, pms.n_avg)

    # Run
    for run in range(pms.n_avg):
        print('### Avg run #'+str(run))
        start_time = time.time()
        launch_training(pms, path, run)
        print("--- %s seconds ---" % (time.time() - start_time))
        filename = path+'/ppo_'+str(run)+'.dat'
        averager.store(filename, run)

    # Write to file
    filename  = path+'/ppo_avg.dat'
    averager.average(filename)

    # Plot
    os.system('gnuplot -c dragonfly/plot/plot.gnu '+path)
