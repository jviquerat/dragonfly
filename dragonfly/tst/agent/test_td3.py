# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test td3 agent
def test_td3():

    runner("dragonfly/tst/agent/td3.json",
           "td3")
