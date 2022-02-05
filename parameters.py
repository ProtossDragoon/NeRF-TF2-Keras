import tensorflow as tf
import numpy as np


class CommonParams():

    # Initialize global variables.
    AUTO = tf.data.AUTOTUNE
    TRAIN_TEST_SPLIT = 0.8

    def __init__(self
    ):
        self.batch_size = 5
        self.epochs = 40


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
        super().__init__()
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

    def get_number_of_flatten_rays(self):
        n = (
            self.image_h 
            * self.image_w 
            * self.n_samples_per_ray
        )
        return n

    def get_dimension_of_network_input_ray(self):
        d = (
            self.n_pos_encoding_fn
            * self.image_ch
            * self.pos_encoding_dims
            + self.image_ch # positional encoding were concatenated on default channel
        )
        return d

