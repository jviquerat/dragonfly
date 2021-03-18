# Generic imports
import os
import sys
import time
import numpy as np

# Custom imports
from dragonfly.core.training import *
from dragonfly.utils.json    import *
from dragonfly.utils.data    import *
from dragonfly.agent.agent   import *
from dragonfly.envs.par_envs import *

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

    # Declare environement
    env   = par_envs(pms.env_name, pms.n_cpu, path)
    agent = agent_factory.create(pms.agent,
                                 obs_dim = env.obs_dim,
                                 act_dim = env.act_dim,
                                 pms     = pms)

    # Intialize averager
    averager = data_avg(agent.n_vars, pms.n_ep, pms.n_avg)

    # Run
    for run in range(pms.n_avg):
        print('### Avg run #'+str(run))
        start_time = time.time()
        agent.reset()
        launch_training(pms, path, run, env, agent)
        dt = time.time() - start_time
        dt = f"{dt:.3f}"
        print("# Elapsed time: "+str(dt)+" seconds")
        filename = path+'/'+pms.agent+'_'+str(run)+'.dat'
        averager.store(filename, run)

    # Close environments
    env.close()

    # Write to file
    filename = path+'/'+pms.agent+'_avg.dat'
    averager.average(filename)

    # Plot
    os.system('gnuplot -c dragonfly/plot/plot.gnu '+path)
