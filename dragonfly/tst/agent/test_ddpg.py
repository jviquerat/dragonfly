# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test ddpg agent
def test_ddpg():

    runner("dragonfly/tst/agent/ddpg.json",
           "ddpg")
