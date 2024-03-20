# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *

###############################################
### Test lstm
def test_lstm():

    runner("dragonfly/tst/network/lstm.json",
           "lstm")
