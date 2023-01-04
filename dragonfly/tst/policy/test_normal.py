# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst           import *
from dragonfly.src.policy.normal import *
from dragonfly.src.utils.json    import *

###############################################
### Test normal policy
def test_normal():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/normal.json")

    # Initialize discrete agent
    policy = normal(1, 5, reader.pms.policy)

    # Test action values
    print("Test normal policy")
    obs = [[1.0]]
    act, lgp = policy.actions(obs)
    print("Actions:",act)

    obs = tf.cast([obs], tf.float32)
    mu, sg = policy.forward(obs)
    assert(np.all(np.abs(mu) < 1.0))
    assert(np.all(np.abs(sg) < 1.0))
    assert(np.all(np.abs(sg) > 0.0))

    print("")

    #########################
    # Initialize json parser and read test json file
    reader.read("dragonfly/tst/policy/normal_log.json")

    # Initialize discrete agent
    policy = normal(1, 5, reader.pms.policy)

    # Test action values
    print("Test normal policy")
    obs = [[1.0]]
    act, lgp = policy.actions(obs)
    print("Actions:",act)

    obs = tf.cast([obs], tf.float32)
    mu, sg = policy.forward(obs)
    assert(np.all(np.abs(mu) <  1.0))
    assert(np.all(np.abs(sg) <= 1.0))
    assert(np.all(np.abs(sg) >  0.0))

    print("")
