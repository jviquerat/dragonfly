# Generic imports
import pytest
import torch
import numpy as np

# Custom imports
from dragonfly.tst.tst import *
from dragonfly.src.policy.normal_iso import *
from dragonfly.src.utils.json import *

###############################################
### Test isotropic normal policy
def test_normal_iso():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/normal_iso.json")

    # Initialize discrete agent
    policy = normal_iso(1, 5, reader.pms.policy)

    # Test action values
    print("Test isotropic normal policy")
    obs = torch.Tensor([[1.0]])
    act, lgp = policy.actions(obs)
    print("Actions:", act)

    mu, sg = policy.forward(obs)
    assert(np.all(np.abs(mu.detach().cpu().numpy()) < 1.0))
    assert(np.all(np.abs(sg.detach().cpu().numpy()) < 1.0))
    assert(np.all(np.abs(sg.detach().cpu().numpy()) > 0.0))

    print("")
