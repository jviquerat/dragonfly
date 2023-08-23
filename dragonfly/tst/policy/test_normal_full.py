# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst                import *
from dragonfly.src.policy.normal_full import *
from dragonfly.src.utils.json         import *

###############################################
### Test full normal policy
def test_normal_full():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/normal_full.json")

    # Initialize discrete agent
    policy = normal_full(1, 5, reader.pms.policy)

    # Test action values
    print("Test full normal policy")
    obs = [[1.0]]
    act, lgp = policy.actions(obs)
    print("Actions:",act)

    obs = tf.cast([obs], tf.float32)
    mu, sg, cr = policy.forward(obs)
    assert(np.all(np.abs(mu) < 1.0))
    assert(np.all(np.abs(sg) < 1.0))
    assert(np.all(np.abs(sg) > 0.0))
    assert(np.all(np.abs(cr) > 0.0))
    assert(np.all(np.abs(cr) < 1.0))

    print("")
