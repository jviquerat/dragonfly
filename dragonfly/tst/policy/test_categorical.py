# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst                import *
from dragonfly.src.policy.categorical import *
from dragonfly.src.utils.json         import *

###############################################
### Test categorical policy
def test_categorical():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/categorical.json")

    # Initialize discrete agent
    policy = categorical(1, 2, reader.pms.policy)

    # Test action values
    print("Test categorical policy")
    obs = [[1.0]]
    act = policy.get_actions(obs)
    print("Actions:",act)

    assert((act == [0]) or (act == [1]))
    print("")
