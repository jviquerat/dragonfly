# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test continuous ppo agent
def test_ppo_continuous():

    runner("dragonfly/tst/agent/ppo_continuous.json",
           "continuous ppo")
