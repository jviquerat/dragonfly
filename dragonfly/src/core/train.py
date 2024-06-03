# Generic imports
import os
import sys
import time
import shutil
import numpy as np

# Custom imports
from dragonfly.src.env.mpi         import *
from dragonfly.src.core.constants  import *
from dragonfly.src.core.paths      import *
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
    paths.base     = os.path.abspath(os.getcwd())
    paths.results  = 'results'
    os.makedirs(paths.results, exist_ok=True)
    paths.results += '/' + folder_name(pms)
    os.makedirs(paths.results, exist_ok=True)

    # Copy json file to results folder
    shutil.copyfile(json_file, paths.results+'/params.json')

    # Intialize averager
    averager = data_avg(2, int(pms.n_stp_max/step_report), pms.n_avg)

    # Initialize trainer
    trainer = trainer_factory.create(pms.trainer.style,
                                     env_pms   = pms.env,
                                     agent_pms = pms.agent,
                                     n_stp_max = pms.n_stp_max,
                                     pms       = pms.trainer)

    # Run
    for run in range(pms.n_avg):
        liner()
        print('Avg run #'+str(run))
        paths.run = paths.results + '/'+str(run)
        os.makedirs(paths.run, exist_ok=True)
        trainer.reset()
        trainer.loop()
        filename = paths.run + '/data.dat'
        averager.store(filename, run)

    # Close environments
    trainer.env.close()

    # Write to file
    filename = paths.results + '/avg.dat'
    data = averager.average(filename)

    # Plot
    name = paths.results + '/' + folder_name(pms)
    plot_avg(data, name)

    # Finalize main process
    mpi.finalize()

# Generate results folder name
def folder_name(pms):

    name_env = ""
    if hasattr(pms.naming, "env"):
        if (pms.naming.env is True):
            name_env = pms.env.name

    name_agent = ""
    if hasattr(pms.naming, "agent"):
        if (pms.naming.agent is True):
            name_agent = pms.agent.type

    name_tag = ""
    if hasattr(pms.naming, "tag"):
        if (pms.naming.tag is not False):
            name_tag = pms.naming.tag

    name_time = ""
    if hasattr(pms.naming, "time"):
        if (pms.naming.time is True):
            name_time = str(time.strftime("%H-%M-%S", time.localtime()))

    path = ""
    if (name_env != ""):
        path += name_env
    if (name_agent != ""):
        if (path != ""): path += "_"
        path += name_agent
    if (name_tag != ""):
        if (path != ""): path += "_"
        path += name_tag
    if (name_time):
        if (path != ""): path += "_"
        path += name_time

    return path
