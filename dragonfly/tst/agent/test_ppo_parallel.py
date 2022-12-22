# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test continuous ppo agent in parallel
def test_ppo_parallel():

    runner("dragonfly/tst/agent/ppo_parallel.json",
           "parallel ppo")
