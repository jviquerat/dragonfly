# Generic imports
import os

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.utils.json      import json_parser
from dragonfly.src.env.environment import environment
from dragonfly.tst.runner          import runner

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
    env  = environment(path, pms.env)

    # Check
    assert(env.worker.env.env.env.env.enable_wind == True)
    assert(env.spaces.obs_clip_  == False)
    assert(env.spaces.obs_norm_  == True)
    assert(env.spaces.obs_noise_ == False)
    assert(env.spaces.obs_stack_ == 4)
    print("")

    #########################
    # Test obs stacking
    runner("dragonfly/tst/env/obs_stack.json",
           "observation stack")
