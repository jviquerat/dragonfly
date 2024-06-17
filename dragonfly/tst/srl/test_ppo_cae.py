# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test ppo agent with cae as srl method
def test_ppo_cae():

    runner("dragonfly/tst/srl/ppo-srl-cae.json",
           "ppo srl cae")
