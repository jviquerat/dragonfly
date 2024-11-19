# Generic imports
import pytest
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.tst.tst                 import *
from dragonfly.src.policy.normal_const import normal_const
from dragonfly.src.utils.json          import json_parser

###############################################
### Test constant normal policy
def test_normal_const():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/normal_const.json")

    # Initialize discrete agent
    policy = normal_const(1, [1], 5, reader.pms.policy)

    # Test action values
    print("Test const normal policy")
    obs = tf.constant([[1.0]])
    act, lgp = policy.actions(obs)
    print("Actions:",act)

    mu = policy.forward(obs)
    assert(np.all(np.abs(mu) < 1.0))

    print("")
