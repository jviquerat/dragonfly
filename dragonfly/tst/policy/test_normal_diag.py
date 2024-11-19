# Generic imports
import pytest
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.tst.tst                import *
from dragonfly.src.policy.normal_diag import normal_diag
from dragonfly.src.utils.json         import json_parser

###############################################
### Test diagonal normal policy
def test_normal_diag():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/normal_diag.json")

    # Initialize discrete agent
    policy = normal_diag(1, [1], 5, reader.pms.policy)

    # Test action values
    print("Test diagonal normal policy")
    obs = tf.constant([[1.0]])
    act, lgp = policy.actions(obs)
    print("Actions:",act)

    mu, sg = policy.forward(obs)
    assert(np.all(np.abs(mu) < 1.0))
    assert(np.all(np.abs(sg) < 1.0))
    assert(np.all(np.abs(sg) > 0.0))

    print("")
