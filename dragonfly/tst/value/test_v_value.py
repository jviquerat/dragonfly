# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst           import *
from dragonfly.src.value.v_value import *
from dragonfly.src.utils.json    import *

###############################################
### Test v_value
def test_v_value():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/value/v_value.json")

    # Initialize v_value
    value = v_value(1, [1], reader.pms.value)

    # Test action values
    print("Test v_value")
    obs = [[1.0]]
    val = value.values(obs)
    print("Value:", val)

    assert(len(val) == 1)
    assert(len(val[0]) == 1)
    print("")
