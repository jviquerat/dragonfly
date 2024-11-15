# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test discrete ppo agent
def test_ppo_discrete():

    runner("dragonfly/tst/agent/ppo_discrete.json",
           "discrete ppo")
