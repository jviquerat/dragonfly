# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test sac agent
def test_sac():

    runner("dragonfly/tst/agent/sac.json",
           "sac")
