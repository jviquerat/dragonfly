# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test ppo agent with pca as srl method
def test_ppo_pca():

    runner("dragonfly/tst/srl/ppo-srl-pca.json",
           "ppo srl pca")
