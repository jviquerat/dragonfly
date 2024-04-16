import tensorflow as tf
import numpy as np
import unittest

from dragonfly.src.utils.rmsnorm import *


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        self.model = RMSNorm()

    def tearDown(self):
        self.model = None

    def test_output_shape(self):
        input_shape = (4, 5, 3)
        inputs = tf.ones(input_shape)
        outputs = self.model(inputs)
        self.assertEqual(outputs.shape, input_shape)

    def test_output_values(self):
        input_data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
        inputs = tf.expand_dims(input_data, axis=0)

        expected_output = tf.constant(
            [[[0.46291006, 0.9258201, 1.3887302], [0.7895421, 0.98692757, 1.184313]]],
            dtype=tf.float32,
        )
        outputs = self.model(inputs)
        tf.debugging.assert_near(outputs, expected_output, atol=1e-5)


if __name__ == "__main__":
    unittest.main()