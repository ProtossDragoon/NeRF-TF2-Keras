# Third party Packages and Modules
import tensorflow as tf
import numpy as np


def get_camera_translation_z(t):
    """Get the z-axis translation matrix for movement in t
    """
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_camera_rotation_pitch(phi):
    """Get the x-axis rotation matrix for movement in phi.
    As same as 'pitch' or 'tilt'.
    """
    matrix = [
        [1, 0, 0, 0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_camera_rotation_yaw(theta):
    """Get the y-axis rotation matrix for movement in theta.
    As same as 'yaw' or 'pan'.
    """
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [            0, 1,              0, 0],
        [tf.sin(theta), 0,  tf.cos(theta), 0],
        [            0, 0,              0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def pose_spherical(*, 
    rotation_yaw:float=None, 
    rotation_pitch:float=None, 
    translation_z:int=None
):
    """
    Get the camera to world matrix for the corresponding 
    theta, phi and z.
    """
    c2w = get_camera_translation_z(translation_z)
    c2w = get_camera_rotation_yaw(rotation_yaw / 180.0 * np.pi) @ c2w
    c2w = get_camera_rotation_pitch(rotation_pitch / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w