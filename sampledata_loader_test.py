# Internal Packages and Modules
import unittest

# Third party Packages and Modules
import numpy as np

# NeRF project
import sampledata_loader
from parameters import NeRFParams


def setUpModule():
    global images, poses, focal_length
    global num_images
    global nerf_params
    # Load numpy formed data.
    images, poses, focal_length = sampledata_loader.get_np_data_from_local_file('./data/tiny_nerf_data.npz')
    num_images, image_h, image_w, _ = images.shape
    # Save params.
    nerf_params = NeRFParams(
        image_h=image_h,
        image_w=image_w,
        focal_length=focal_length,
    )


class TestDataLoader(unittest.TestCase):

    def test_get_data_from_local_npz_file(self):
        self.assertEqual(len(images[0].shape), 3, 'Image data should be [h,w,ch] form.')
        self.assertEqual(tuple(poses[0].shape), (4,4,), 'Pose matrix (camera extrinsic) should be [4,4] shaped.')
        self.assertEqual(tuple(focal_length.shape), (), 'Focal length should be a constant.')

    def test_convert_to_tensorflow_dataset_format(self):
        train_ds, val_ds = sampledata_loader.get_train_val_tf_ds(
            images, 
            poses, 
            nerf_params
        )
        train_it = iter(train_ds)
        self.assertEqual(next(train_it)[0].shape[0], nerf_params.batch_size)
        val_it = iter(val_ds)
        self.assertEqual(next(val_it)[0].shape[0], nerf_params.batch_size)


if __name__ == '__main__':
    unittest.main()