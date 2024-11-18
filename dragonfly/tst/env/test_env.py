# Generic imports
import os

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.utils.json      import json_parser
from dragonfly.src.env.environment import environment
from dragonfly.tst.runner          import runner

###############################################
### Test passing arguments to environment
def test_env_args():

    # Initial space
    print("")

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

###############################################
### Test observation stacking
def test_obs_stacking():

    runner("dragonfly/tst/env/obs_stack.json",
           "observation stack")

###############################################
### Test 2D obs downscale
def test_obs_2D_downscale():

    runner("dragonfly/tst/env/obs_downscale.json",
           "observation downscale")

###############################################
### Test 2D obs grayscale
def test_obs_2D_grayscale():

    runner("dragonfly/tst/env/obs_grayscale.json",
           "observation grayscale")

###############################################
### Test 2D obs downscale + grayscale
def test_obs_2D_downscale_grayscale():

    runner("dragonfly/tst/env/obs_downscale_grayscale.json",
           "observation downscale and grayscale")

###############################################
### Test 2D obs frameskip
def test_obs_2D_frameskip():

    runner("dragonfly/tst/env/obs_frameskip.json",
           "observation frameskip")
