# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test dqn agent
def test_dqn():

    runner("dragonfly/tst/agent/dqn.json",
           "dqn")
