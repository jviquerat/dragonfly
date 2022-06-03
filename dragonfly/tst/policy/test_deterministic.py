# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst                  import *
from dragonfly.src.policy.deterministic import *
from dragonfly.src.utils.json           import *

###############################################
### Test deterministic policy
def test_deterministic():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/deterministic.json")

    # Initialize discrete agent
    policy = deterministic(1, 5, reader.pms.policy)

    # Test action values
    print("Test deterministic policy")
    obs = [[1.0]]
    act = policy.actions(obs)
    print("Actions:",act)

    obs = tf.cast([obs], tf.float32)
    out = policy.forward(obs)

    print("")
