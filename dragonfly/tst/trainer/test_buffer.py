# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import runner

###############################################
### Test buffer-based training
def test_buffer():

    runner("dragonfly/tst/trainer/buffer.json",
           "buffer-based trainer")
