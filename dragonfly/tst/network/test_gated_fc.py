# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test d2rl
def test_gated_fc():

    runner("dragonfly/tst/network/gated_fc.json",
           "gated_fc")
