# Internal Packages and Modules
import unittest

# NeRF project
import simple_dataloader
from parameters import NeRFParams
from nerf_core import architecture
from nerf_core import nerf
from nerf_core import optimizer
from nerf_core import loss


def setUpModule():
    global images, poses, focal_length
    global num_images
    global nerf_params
    # Load numpy formed data.
    images, poses, focal_length = simple_dataloader.get_np_data_from_local_file('./tiny_nerf_data.npz')
    num_images, image_h, image_w, _ = images.shape
    # Save params.
    nerf_params = NeRFParams(
        image_h=image_h,
        image_w=image_w,
        focal_length=focal_length,
    )


class SmallDatasetShortTraining(unittest.TestCase):
    
    def setUp(self):
        nerf_params.epochs = 2

    def test_short_training(self):
        train_ds, val_ds = simple_dataloader.get_train_val_tf_ds(images, poses, nerf_params)
        nerf_architecture = architecture.DNNArchitecture(nerf_params, n_layers=8).get_nerf_architecture()
        nerf_architecture.summary()

        nerf_model = nerf.NeRFModel(nerf_architecture, nerf_params)
        nerf_model.compile(
            optimizer=optimizer.get_adam_optimizer(),
            loss_fn=loss.get_mse_loss(),
        )

        sample_batched_test_imgs, sample_batched_test_rays = next(iter(train_ds))
        sample_batched_test_rays_flat, sample_batched_test_ray_t = sample_batched_test_rays

        # build shape
        nerf_model(sample_batched_test_rays)
        nerf_model.summary()

        nerf_model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=nerf_params.batch_size,
            epochs=nerf_params.epochs,
            steps_per_epoch=(num_images * nerf_params.TRAIN_TEST_SPLIT) // nerf_params.batch_size,
        )


if __name__ == '__main__':
    unittest.main()