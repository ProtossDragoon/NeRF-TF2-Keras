# Third party Packages and Modules
import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm

# NeRF project
import utils.transform as transform
import nerf_core.ray as ray
from parameters import NeRFParams
from nerf_core.nerf import NeRFModel 


def hemesphere_ray_batch_generator(
    nerf_params:NeRFParams,
):
    # Iterate over different theta value and generate scenes.
    for yaw in tqdm(np.linspace(0.0, 360.0, 120, endpoint=False)):
        # Get the camera to world matrix.
        c2w = transform.pose_spherical(
            rotation_yaw=yaw,
            rotation_pitch=-30.0,
            translation_z=4.0,
        )
        (rays_flat, ray_t) = ray.pose_to_ray(
            pose=c2w,
            image_h=nerf_params.image_h,
            image_w=nerf_params.image_w,
            focal_length=nerf_params.focal_length,
            n_samples_per_ray=nerf_params.n_samples_per_ray,
            pos_encoding_dims=nerf_params.pos_encoding_dims
        )
        yield (rays_flat, ray_t)


def get_hemesphere_ray_tfds(
    nerf_params:NeRFParams
):
    #NOTE: Unstable TPU Compatibiltiy
    num = nerf_params.get_number_of_flatten_rays()
    dim = nerf_params.get_dimension_of_network_input_ray()
    h = nerf_params.image_h
    w = nerf_params.image_w
    n = nerf_params.n_samples_per_ray
    dataset = tf.data.Dataset.from_generator(
        lambda: hemesphere_ray_batch_generator(nerf_params),
        output_signature=(
            tf.TensorSpec(shape=(num, dim), dtype=tf.float32,), # rays_flat
            tf.TensorSpec(shape=(h, w, n), dtype=tf.float32,), # ray_t
        )
    ).batch(nerf_params.batch_size)
    return dataset


def get_rendered_images(
    nerf_model:NeRFModel,
):
    images = []
    nerf_params = nerf_model.nerf_params
    tfds = get_hemesphere_ray_tfds(nerf_params)
    for (batched_rays_flat, batched_ray_t) in iter(tfds):
        inferenced_batched_rgb, _ = nerf_model.render_rgb_depth(
            batched_rays_flat, batched_ray_t,
        )
        for rgb in inferenced_batched_rgb:
            rgb = np.clip(255*rgb, 0.0, 255.0).astype(np.uint8)
            images.append(rgb)
    return images


def write_video(
    nerf_model:NeRFModel,
    save_dir:str='./result_videos',
    video_name:str='result.mp4',
):
    video_fullpath = tf.io.gfile.join(save_dir, video_name)
    tf.io.gfile.makedirs(save_dir)
    images = get_rendered_images(nerf_model)
    imageio.mimwrite(
        video_fullpath, images,
        fps=30, quality=7, macro_block_size=None
    )


if __name__ == '__main__':
    pass