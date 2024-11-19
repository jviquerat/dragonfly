# Generic imports
import pytest
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.tst.tst                import *
from dragonfly.src.policy.tanh_normal import tanh_normal
from dragonfly.src.utils.json         import json_parser

###############################################
### Test tanh-normal policy
def test_normal():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/tanh_normal.json")

    # Initialize discrete agent
    policy = tanh_normal(1, [1], 5, reader.pms.policy)

    # Test action values
    print("Test tanh-normal policy")
    obs = tf.constant([[1.0]])
    act, lgp = policy.actions(obs)
    print("Actions:",act)

    tanh_act, lgp = policy.sample(obs)
    assert(np.all(np.abs(tanh_act) < 1.0))

    print("")
