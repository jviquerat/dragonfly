# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test conv1d
def test_conv1d():

    runner("dragonfly/tst/network/conv1d.json",
           "conv1d")
