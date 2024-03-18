# Generic imports
import os

# Custom imports
from dragonfly.src.utils.json       import *
from dragonfly.src.env.environments import *
from dragonfly.tst.tst              import *
from dragonfly.tst.runner           import *

###############################################
### Test environment interface
def test_env_args():

    # Initial space
    print("")

    #########################
    # Test environment optional arguments
    reader = json_parser()
    pms    = reader.read("dragonfly/tst/env/env_args.json")

    # Initialize environment
    path = os.path.abspath(os.getcwd())
    env  = environments(path, pms.env)

    # Check
    assert(env.worker.env.enable_wind == True)
    assert(env.obs_clip  == False)
    assert(env.obs_norm  == True)
    assert(env.obs_noise == False)
    assert(env.obs_stack == 4)
    print("")

    #########################
    # Test obs stacking
    runner("dragonfly/tst/env/obs_stack.json",
           "observation stack")
