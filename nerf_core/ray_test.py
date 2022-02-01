# Internal Packages and Modules
import unittest

# Third party Packages and Modules
import numpy as np

# NeRF project
from nerf_core import ray


class TestRayGenerator(unittest.TestCase):

    def test_pose_to_ray(self):
        pose = np.ones([4, 4])
        ray.pose_to_ray(
            pose,
            image_h=100,
            image_w=100,
            focal_length=20,
            n_samples_per_ray=10,
            pos_encoding_dims=15,
        )

    def test_encode_position(self):
        x = np.ones([5, 3], dtype=np.float32)
        ray.encode_position(
            x,
            pos_encoding_dims=3,
        )
    

if __name__ == '__main__':
    unittest.main()