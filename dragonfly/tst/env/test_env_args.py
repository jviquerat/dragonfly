# Generic imports
import os

# Custom imports
from dragonfly.src.utils.json       import *
from dragonfly.src.env.environments import *

###############################################
### Test environment with optional arguments
def test_env_args():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    pms    = reader.read("dragonfly/tst/env/env_args.json")

    # Initialize environment
    path = os.path.abspath(os.getcwd())
    env  = environments(path, pms.env)

    # Check
    assert(env.worker.env.enable_wind == True)
    print("")
