from dragonfly.src.utils.similarity import *
from dragonfly.src.utils.similarity import _normalize, _find_similar_states_indexes, _get_similar_states
import unittest

class TestGetUpgradedStates(unittest.TestCase):
    def setUp(self):
        self.full_obs = tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32)
        self.batch_obs = tf.constant([[2, 3], [6, 7]], dtype=tf.float32)
        self.info = (
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32),
            tf.constant([[1, 2], [10, 10], [10, 10], [7, 8]], dtype=tf.float32)
        )
        self.max_distance = 2.0

    def test_normalize(self):
        normalized_obs = _normalize(self.full_obs)
        assert tf.linalg.normalize(normalized_obs)[1] == 1.0
   
    def test_find_similar_states_indexes(self):
        similar_states_indexes = _find_similar_states_indexes(self.full_obs, self.batch_obs, self.max_distance)
        self.assertTrue(np.array_equal(similar_states_indexes, np.array([0, 3], dtype=np.int32)))

    def test_get_similar_states(self):
        upgraded_states = _get_similar_states(self.full_obs, self.batch_obs, self.info, self.max_distance)
        print(upgraded_states)
        expected_output = (tf.constant([[1, 2], [7, 8]], dtype=tf.float32),tf.constant([[1, 2], [7, 8]], dtype=tf.float32),tf.constant([[1, 2], [7, 8]], dtype=tf.float32),tf.constant([[1, 2], [7, 8]], dtype=tf.float32))
        for i, tensor in enumerate(upgraded_states):
            self.assertTrue(tf.reduce_all(tf.math.equal(tensor, expected_output[i])).numpy())

    def test_get_upgraded_states(self):
        upgraded_states = get_upgraded_states(self.full_obs, 1, 3, self.info, self.max_distance)
        expected_output = (
            tf.constant([[10, 10], [10, 10], [1, 2], [7, 8]], dtype=tf.float32),
            tf.constant([[10, 10], [10, 10], [1, 2], [7, 8]], dtype=tf.float32),
            tf.constant([[10, 10], [10, 10], [1, 2], [7, 8]], dtype=tf.float32),
            tf.constant([[10, 10], [10, 10], [1, 2], [7, 8]], dtype=tf.float32)
        )
        for i, tensor in enumerate(upgraded_states):
            self.assertTrue(tf.reduce_all(tf.math.equal(tensor, expected_output[i])).numpy())