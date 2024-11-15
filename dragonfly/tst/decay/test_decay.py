# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst          import *
from dragonfly.src.decay.linear import linear
from dragonfly.src.utils.json   import json_parser

###############################################
### Test decay class
def test_decay():

    # Initial space
    print("")

    #########################
    # Test linear decay
    print("Linear decay")

    # Read json file and declare decay
    reader = json_parser()
    reader.read("dragonfly/tst/decay/linear.json")
    epsilon = linear(reader.pms.decay)

    # Test initial value
    val = epsilon.get()
    print("Initial value:", val)
    assert(val==1.0)

    # Decay
    for i in range(200):
        epsilon.decay()

    # Test minimal value
    val = epsilon.get()
    print("Final value:", val)
    assert(val==0.0)

    print("")
