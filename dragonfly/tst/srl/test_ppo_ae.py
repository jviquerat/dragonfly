# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test ppo agent with ae as srl method
def test_ppo_ae():

    runner("dragonfly/tst/srl/ppo-srl-ae.json",
           "ppo srl ae")
