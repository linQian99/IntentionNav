"""Small geometry helpers used by the IntentionNav collection pipeline."""

import numpy as np


DEFAULT_CAMERA_FORWARD = np.array([1, 0, 0])


def rot3_from_o_to_ab(origin_forward, point_a, point_b):
    """Return a 3x3 rotation that points ``origin_forward`` from A toward B."""
    v = origin_forward / np.linalg.norm(origin_forward)
    vec_ab = point_b - point_a
    if np.linalg.norm(vec_ab) < 1e-6:
        return np.eye(3)

    w = vec_ab / np.linalg.norm(vec_ab)
    dot = np.dot(v, w)

    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        axis = np.cross(v, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        k_mat = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + 2 * (k_mat @ k_mat)

    axis = np.cross(v, w)
    axis /= np.linalg.norm(axis)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    k_mat = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * k_mat + (1 - np.cos(theta)) * (k_mat @ k_mat)
