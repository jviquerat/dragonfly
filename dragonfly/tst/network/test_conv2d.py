# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test cnn
def test_conv2d():

    runner("dragonfly/tst/network/conv2d.json",
           "conv2d")
