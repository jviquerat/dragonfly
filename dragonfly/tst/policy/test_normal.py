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
    act, lgp = policy.get_actions(obs)
    print("Actions:",act)

    assert(np.all(act <= 1.0))
    assert(np.all(act >=-1.0))
    print("")
