# Generic imports
import pytest
import torch
import numpy as np

# Custom imports
from dragonfly.tst.tst import *
from dragonfly.src.policy.tanh_normal import *
from dragonfly.src.utils.json import *

###############################################
### Test tanh-normal policy
def test_normal():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/policy/tanh_normal.json")

    # Initialize discrete agent
    policy = tanh_normal(1, 5, reader.pms.policy)

    # Test action values
    print("Test tanh-normal policy")
    obs = torch.Tensor([[1.0]])
    act, lgp = policy.actions(obs)
    print("Actions:", act)

    tanh_act, lgp = policy.sample(obs)
    assert(np.all(np.abs(tanh_act.detach().cpu().numpy()) < 1.0))

    print("")
