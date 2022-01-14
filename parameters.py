import tensorflow as tf
import numpy as np


class CommonParams():
    # Initialize global variables.
    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = 5
    EPOCHS = 40
    TRAIN_TEST_SPLIT = 0.8


class NeRFParams(CommonParams):

    def __init__(self,
        image_h:int,
        image_w:int,
        focal_length:float, *,
        fixed_camera_intrinsic:np.ndarray=None,
        pos_encoding_dims:int=16,
        n_samples_per_ray:int=32,
        image_ch:int=3,
    ):
        if image_ch != 3:
            raise NotImplementedError('Other cases were not implemented.')

        self.image_h = image_h
        self.image_w = image_w
        self.image_ch = image_ch
        self.focal_length = focal_length
        self.camera_intrinsic = fixed_camera_intrinsic
        self.pos_encoding_dims = pos_encoding_dims
        self.n_samples_per_ray = n_samples_per_ray
        self.n_pos_encoding_fn = 2 # sin, cos