# Generic imports
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.src.utils.rmsnorm import *

###############################################
### Test rmsnorm shape
def test_rmsnorm_shape():

    model       = RMSNorm()
    input_shape = (4, 5, 3)
    inputs      = tf.ones(input_shape)
    outputs     = model(inputs)

    assert outputs.shape == input_shape

### Test rmsnorm output
def test_rmsnorm():

    model      = RMSNorm()
    input_data = tf.constant([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]], dtype=tf.float32)
    inputs     = tf.expand_dims(input_data, axis=0)

    expected_output = tf.constant(
        [[[0.46291006, 0.9258201, 1.3887302],
          [0.7895421, 0.98692757, 1.184313]]], dtype=tf.float32)
    outputs = model(inputs)

    assert(np.isclose(inputs.numpy().all(),
                      outputs.numpy().all(), atol=1.0e-5))
