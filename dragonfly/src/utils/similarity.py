import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
from typing import Tuple


def _normalize(x: tf.Tensor):
    """Normalizing 2D vectors let us use euclidian distance instead of cosine similarity
    https://medium.com/ai-for-real/relationship-between-cosine-similarity-and-euclidean-distance-7e283a277dff
    """
    return tf.linalg.normalize(x)[0]


def _find_similar_states_indexes(
    full_obs: tf.Tensor, mini_batch: tf.Tensor, max_distance: float = 0.1
):
    full_tree = cKDTree(full_obs)
    batch_tree = cKDTree(mini_batch)

    indexes = batch_tree.query_ball_tree(full_tree, r=max_distance)

    # Flatten and get unique indexes
    unique_indexes = np.unique(np.concatenate(indexes))

    return np.ndarray.astype(unique_indexes, np.int32)


def _find_similar_states_indexes_tf(
    full_obs: tf.Tensor, mini_batch: tf.Tensor, max_distance: float = 0.98
):
    # Calculate norms
    full_norm = tf.norm(full_obs, axis=1, keepdims=True)
    mini_batch_norm = tf.norm(mini_batch, axis=1, keepdims=True)

    # Compute cosine similarity
    dot_product = tf.matmul(full_obs, mini_batch, transpose_b=True)
    cosine_sim = dot_product / (full_norm * tf.transpose(mini_batch_norm))

    # Find indexes where cosine similarity exceeds max_distance
    similar_indexes = tf.where(cosine_sim > max_distance)

    unique_indexes, _ = tf.unique(similar_indexes[:, 0])
    return unique_indexes


def _get_similar_states(
    full_obs: tf.Tensor,
    mini_batch: tf.Tensor,
    info: Tuple[tf.Tensor],
    max_distance: float = 0.1,
    use_ckdtree: bool = True,
) -> Tuple[tf.Tensor]:
    if use_ckdtree:
        similar_states_indexes = tf.numpy_function(
            _find_similar_states_indexes,
            [full_obs, mini_batch, max_distance],
            [tf.int32],
        )
    else:
        similar_states_indexes = _find_similar_states_indexes_tf(
            full_obs, mini_batch, max_distance
        )
    return tuple(tf.gather(x, similar_states_indexes) for x in info)


def get_upgraded_states(
    obs: tf.Tensor,
    start: int,
    end: int,
    info: Tuple[tf.Tensor],
    max_distance: float = 0.1,
    use_ckdtree: bool = True,
) -> Tuple[tf.Tensor]:
    """
    Get upgraded states by finding similar states within a maximum distance.

    Args:
        obs (tf.Tensor): The full observation tensor.
        start (int): Start index for the batch.
        end (int): End index for the batch.
        info (Tuple[tf.Tensor]): Additional information tensors, such as (obs, act, adv, ...).
        max_distance (float): Maximum distance to consider for similarity.

    Returns:
        Tuple[tf.Tensor]: Upgraded states based on similarity.
    """
    full_obs = tf.concat([obs[:start], obs[end:]], 0)
    batch_obs = obs[start:end]

    _info = tuple(tf.concat([a[:start], a[end:]], 0) for a in info)

    info_to_add = _get_similar_states(
        full_obs=_normalize(full_obs),
        mini_batch=_normalize(batch_obs),
        info=_info,
        max_distance=max_distance,
        use_ckdtree=use_ckdtree,
    )

    return (tf.concat([a[start:end], b], 0) for a, b in zip(info, info_to_add))
