# Internal Packages and Modules
import unittest

# Third party Packages and Modules
import tensorflow as tf

# NeRF project
import sampledata_loader
from parameters import NeRFParams
from nerf_core import architecture
from nerf_core import nerf
from utils import videovis


def setUpModule():
    global nerf_model, nerf_params
    global rays_flat_shape, ray_t_shape
    images, poses, focal_length = sampledata_loader.get_np_data_from_local_file('./data/tiny_nerf_data.npz')
    num_images, image_h, image_w, _ = images.shape
    nerf_params = NeRFParams(
        image_h=image_h,
        image_w=image_w,
        focal_length=focal_length,
    )
    nerf_architecture = architecture.DNNArchitecture(nerf_params, n_layers=5).get_nerf_architecture()
    nerf_model = nerf.NeRFModel(nerf_architecture, nerf_params)

    num = nerf_params.get_number_of_flatten_rays()
    dim = nerf_params.get_dimension_of_network_input_ray()
    rays_flat_shape = (num, dim)
    ray_t_shape = (image_h, image_w, nerf_params.n_samples_per_ray)


class VideoVisualizer(unittest.TestCase):
    
    def test_hemesphere_ray_batch_generator(self):
        print(f'\nDesired rays_flat shape: {rays_flat_shape}')
        print(  f'Desired ray_t shape: {ray_t_shape}')
        gen = videovis.hemesphere_ray_batch_generator(nerf_params, -30)
        for i, (rays_flat, ray_t) in enumerate(gen):
            self.assertEqual(tuple(rays_flat.shape), rays_flat_shape)
            self.assertEqual(tuple(ray_t.shape), ray_t_shape)

    def test_get_hemesphere_ray_tfds(self):
        pass

    def test_get_rendered_images(self):
        pass

    def test_write_video(self):
        global save_dir, video_name
        save_dir='./for_test_temp_dir'
        video_name='for_test_temp_vid.mp4'
        videovis.write_video(
            nerf_model=nerf_model,
            save_dir=save_dir,
            video_name=video_name,
        )

    def test_write_multiple_videos(self):
        global save_dir, video_name
        save_dir='./for_test_temp_dir'
        video_name='for_test_temp_vid.mp4'
        videovis.write_multiple_videos(
            nerf_model=nerf_model,
            save_dir=save_dir,
            video_name=video_name,
        )


def tearDownModule():
    tf.io.gfile.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()