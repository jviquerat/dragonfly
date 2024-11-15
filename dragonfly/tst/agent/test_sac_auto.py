# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test sac_auto agent
def test_sac_auto():

    runner("dragonfly/tst/agent/sac_auto.json",
           "sac_auto")
