# Generic imports
import numpy as np
import torch
import torch.nn as nn

# Custom imports
from dragonfly.src.utils.rmsnorm import *

###############################################
### Test rmsnorm shape
def test_rmsnorm_shape():
    model = RMSNorm(3)
    input_shape = (4, 5, 3)
    inputs = torch.ones(input_shape)
    outputs = model(inputs)

    assert outputs.shape == input_shape

### Test rmsnorm output
def test_rmsnorm():
    model = RMSNorm(3)
    input_data = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]], dtype=torch.float32)
    inputs = input_data.unsqueeze(0)

    expected_output = torch.tensor(
        [[[0.4629, 0.9258, 1.3887],
         [0.7895, 0.9869, 1.1843]]], dtype=torch.float32)
    outputs = model(inputs)

    assert np.isclose(expected_output.detach().numpy(),
                      outputs.detach().numpy(),
                      atol=1.0e-3).all()
