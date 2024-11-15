# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test temporal-difference training
def test_td():

    runner("dragonfly/tst/trainer/td.json",
           "temporal-difference trainer")
