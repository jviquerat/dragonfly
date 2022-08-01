# Generic imports
import os
import sys
import time
import shutil
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
    base_path     = os.path.abspath(os.getcwd())
    results_path  = 'results'
    os.makedirs(results_path, exist_ok=True)
    path          = folder_name(pms)
    results_path += '/'+path
    os.makedirs(results_path, exist_ok=True)

    # Copy json file to results folder
    shutil.copyfile(json_file, results_path+'/params.json')

    # Intialize averager
    averager = data_avg(4, pms.n_ep_max, pms.n_avg)

    # Initialize trainer
    trainer = trainer_factory.create(pms.trainer.style,
                                     env_pms   = pms.env,
                                     agent_pms = pms.agent,
                                     path      = base_path,
                                     n_cpu     = pms.n_cpu,
                                     n_ep_max  = pms.n_ep_max,
                                     pms       = pms.trainer)

    # Run
    for run in range(pms.n_avg):
        liner()
        print('Avg run #'+str(run))
        os.makedirs(results_path+'/'+str(run), exist_ok=True)
        trainer.reset()
        trainer.loop(results_path, run)
        filename = results_path+'/'+str(run)+'/'+str(run)+'.dat'
        averager.store(filename, run)

    # Close environments
    trainer.env.close()

    # Write to file
    filename = results_path+'/avg.dat'
    data = averager.average(filename)

    # Plot
    filename = results_path+'/'+path
    plot_avg(data, filename)

# Generate results folder name
def folder_name(pms):

    name_env = True
    if hasattr(pms, "name_env"): name_env = pms.name_env

    name_agent = True
    if hasattr(pms, "name_agent"): name_agent = pms.name_agent

    name_tag = ""
    if hasattr(pms, "name_tag"): name_tag = pms.name_tag

    name_time = False
    if hasattr(pms, "name_time"): name_time = pms.name_time

    path_time = ""
    if (name_time):
        path_time = str(time.strftime("%H-%M-%S", time.localtime()))

    path = ""
    if (name_env): path += pms.env.name
    if (name_agent != ""):
        if (path != ""): path += "_"
        path += pms.agent.type
    if (name_tag != ""):
        if (path != ""): path += "_"
        path += name_tag
    if (name_time):
        if (path != ""): path += "_"
        path += path_time

    return path
