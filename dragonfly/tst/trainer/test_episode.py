# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst    import *
from dragonfly.tst.runner import *
###############################################
### Test episode-based training
def test_episode():

    runner("dragonfly/tst/trainer/episode.json",
           "episode-based trainer")
