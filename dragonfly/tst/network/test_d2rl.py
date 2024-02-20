# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test d2rl
def test_d2rl():

    runner("dragonfly/tst/network/d2rl.json",
           "d2rl")
