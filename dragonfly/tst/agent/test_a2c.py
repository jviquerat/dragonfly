# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test continuous a2c agent
def test_a2c_continuous():

    runner("dragonfly/tst/agent/a2c.json",
           "continuous a2c")
