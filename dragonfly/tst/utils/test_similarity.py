from dragonfly.src.utils.similarity import *
from dragonfly.src.utils.similarity import (
    _find_similar_states_indexes_tf,
)
import unittest
import pytest
import numpy as np 

class TestGetUpgradedStates(unittest.TestCase):
    def setUp(self):
        self.full_obs = tf.constant(
            [[-1, -2], [10, 10], [10, 10], [-70, -80]], dtype=tf.float32
        )
        self.batch_obs = tf.constant([[-2, -3], [-70, -80]], dtype=tf.float32)
        self.info = (
            tf.constant([[-1, -2], [10, 10], [10, 10], [-70, -80]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
        )
        self.max_distance = 0.99

    def test_find_similar_states_indexes(self):
        similar_states_indexes = _find_similar_states_indexes_tf(
            self.full_obs, self.batch_obs, self.max_distance
        )
        self.assertTrue(
            np.array_equal(similar_states_indexes, np.array([0, 3], dtype=np.int32))
        )

    def test_get_upgraded_states(self):
        full_obs = tf.constant(
            [[-1, -2], [10, 10], [10, 10], [9, 10]], dtype=tf.float32
        )
        info = (
            tf.constant([[-1, -2], [10, 10], [10, 10], [9, 10]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
        )
        upgraded_states = get_upgraded_states(full_obs, 1, 3, info, self.max_distance)
        expected_output = (
            tf.constant([[10, 10], [10, 10], [9, 10]], dtype=tf.float32),
            tf.constant([[10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[10, 10], [10, 10], [7, 8]], dtype=tf.float32),
        )
        print(upgraded_states)
        for i, tensor in enumerate(upgraded_states):
            tf.print(tensor)
            self.assertTrue(
                tf.reduce_all(tf.math.equal(tensor, expected_output[i])).numpy()
            )
