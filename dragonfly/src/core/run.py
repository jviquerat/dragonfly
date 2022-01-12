# Generic imports
import os
import sys
import time
import numpy as np

# Custom imports
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.utils.prints    import *
from dragonfly.src.agent.agent     import *
from dragonfly.src.trainer.trainer import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.plot.plot       import *

# Average training over multiple runs
def run():

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
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(path,     exist_ok=True)

    # Declare environement
    env   = par_envs(pms.env_name, pms.n_cpu, path)
    agent = agent_factory.create(pms.agent.type,
                                 obs_dim = env.obs_dim,
                                 act_dim = env.act_dim,
                                 n_cpu   = pms.n_cpu,
                                 pms     = pms.agent)

    # Intialize averager
    averager = data_avg(4, pms.n_ep_max, pms.n_avg)

    # Initialize training style
    trainer = trainer_factory.create(pms.trainer.style,
                                     obs_dim  = env.obs_dim,
                                     act_dim  = env.act_dim,
                                     pol_dim  = agent.pol_dim,
                                     n_cpu    = pms.n_cpu,
                                     n_ep_max = pms.n_ep_max,
                                     pms      = pms.trainer)

    # Run
    disclaimer()
    for run in range(pms.n_avg):
        liner()
        print('Avg run #'+str(run))
        agent.reset()
        trainer.reset()
        env.set_cpus()
        trainer.loop(path, run, env, agent)
        filename = path+'/'+pms.agent.type+'_'+str(run)+'.dat'
        averager.store(filename, run)

    # Close environments
    env.close()

    # Write to file
    filename = path+'/'+pms.agent.type+'_avg.dat'
    data = averager.average(filename)

    # Plot
    filename = agent.name + ' - ' + pms.env_name
    plot_avg(data, filename)
