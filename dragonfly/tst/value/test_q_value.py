# Generic imports
import pytest
import tensorflow as tf

# Custom imports
from dragonfly.tst.tst           import *
from dragonfly.src.value.q_value import q_value
from dragonfly.src.utils.json    import json_parser

###############################################
### Test q_value
def test_q_value():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/value/q_value.json")

    # Initialize q_value
    value = q_value(1, [1], 4, reader.pms.value)

    # Test action values
    print("Test q_value")
    obs = tf.constant([[1.0]])
    vals = value.forward(obs)
    print("Values:", vals)

    assert(len(vals) == 1)
    assert(len(vals[0]) == 4)
    print("")
