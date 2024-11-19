# Generic imports
import pytest
import tensorflow as tf

# Custom imports
from dragonfly.tst.tst           import *
from dragonfly.src.value.v_value import v_value
from dragonfly.src.utils.json    import json_parser

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
    obs = tf.constant([[1.0]])
    val = value.forward(obs)
    print("Value:", val)

    assert(len(val) == 1)
    assert(len(val[0]) == 1)
    print("")
