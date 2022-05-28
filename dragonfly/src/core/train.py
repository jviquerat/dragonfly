# Generic imports
import os
import sys
import time
import numpy as np

# Custom imports
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.utils.prints    import *
from dragonfly.src.trainer.trainer import *
from dragonfly.src.plot.plot       import *

# Average training over multiple runs
def train(json_file):

    # Initialize json parser and read parameters
    parser = json_parser()
    pms    = parser.read(json_file)

    # Create paths for results and open repositories
    base_path = os.path.abspath(os.getcwd())
    res_path  = 'results'
    path_time = time.strftime("%H-%M-%S", time.localtime())
    path      = res_path+'/'+pms.env_name+'_'+str(path_time)

    # Open repositories
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(path,     exist_ok=True)

    # Intialize averager
    averager = data_avg(4, pms.n_ep_max, pms.n_avg)

    # Initialize trainer
    trainer = trainer_factory.create(pms.trainer.style,
                                     env_name  = pms.env_name,
                                     agent_pms = pms.agent,
                                     path      = base_path,
                                     n_cpu     = pms.n_cpu,
                                     n_ep_max  = pms.n_ep_max,
                                     pms       = pms.trainer)

    # Run
    for run in range(pms.n_avg):
        liner()
        print('Avg run #'+str(run))
        os.makedirs(path+'/'+str(run), exist_ok=True)
        trainer.reset()
        trainer.loop(path, run)
        filename = path+'/'+str(run)+'/'+pms.agent.type+'_'+str(run)+'.dat'
        averager.store(filename, run)

    # Close environments
    trainer.env.close()

    # Write to file
    filename = path+'/'+pms.agent.type+'_avg.dat'
    data = averager.average(filename)

    # Plot
    filename = path+'/'+pms.agent.type + ' - ' + pms.env_name
    plot_avg(data, filename)
