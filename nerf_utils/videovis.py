# Internal Module
import os
import logging
logging.basicConfig()
logger = logging.getLogger('videovis.py')
logger.setLevel(logging.INFO)

# Third party Packages and Modules
import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm

# NeRF project
import nerf_utils.transform as transform
import nerf_core.ray as ray
from parameters import NeRFParams
from nerf_core.nerf import NeRFModel 


def hemesphere_ray_batch_generator(
    nerf_params:NeRFParams,
    rotation_pitch:int,
):
    # Iterate over different theta value and generate scenes.
    for yaw in tqdm(np.linspace(0.0, 360.0, 120, endpoint=False)):
        # Get the camera to world matrix.
        c2w = transform.pose_spherical(
            rotation_yaw=yaw,
            rotation_pitch=rotation_pitch,
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
    nerf_params:NeRFParams,
    rotation_pitch:int,
):
    #NOTE: Unstable TPU Compatibiltiy
    num = nerf_params.get_number_of_flatten_rays()
    dim = nerf_params.get_dimension_of_network_input_ray()
    h = nerf_params.image_h
    w = nerf_params.image_w
    n = nerf_params.n_samples_per_ray
    dataset = tf.data.Dataset.from_generator(
        lambda: hemesphere_ray_batch_generator(
            nerf_params, 
            rotation_pitch=rotation_pitch
        ),
        output_signature=(
            tf.TensorSpec(shape=(num, dim), dtype=tf.float32,), # rays_flat
            tf.TensorSpec(shape=(h, w, n), dtype=tf.float32,), # ray_t
        )
    ).batch(nerf_params.batch_size)
    return dataset


def get_rendered_images(
    nerf_model:NeRFModel,
    rotation_pitch:int=-30,
):
    images = []
    nerf_params = nerf_model.nerf_params
    tfds = get_hemesphere_ray_tfds(nerf_params, rotation_pitch=rotation_pitch)
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
    rotation_pitch:int=None,
):
    def rename(video_name, rotation_pitch):
        ret = video_name
        if rotation_pitch is not None:
            name, extend = os.path.splitext(video_name)
            ret = f'{name}_{rotation_pitch}{extend}'
        return ret

    video_name = rename(video_name, rotation_pitch)
    video_fullpath = tf.io.gfile.join(save_dir, video_name)
    tf.io.gfile.makedirs(save_dir)
    logger.info(f'Create video. Path: {video_fullpath}')

    def get_kwargs():
        kwargs = {}
        if rotation_pitch is not None:
            kwargs['rotation_pitch'] = rotation_pitch
        return kwargs

    images = get_rendered_images(nerf_model, **get_kwargs())
    imageio.mimwrite(
        video_fullpath, images,
        fps=30, quality=7, macro_block_size=None
    )


def write_multiple_videos(
    nerf_model:NeRFModel,
    pitchs:list=[-30, -10, 10, 30],
    **kwargs,
):
    def get_kwargs():
        return kwargs

    for i in pitchs:
        write_video(
            nerf_model=nerf_model,
            rotation_pitch=i,
            **get_kwargs(),
        )


if __name__ == '__main__':
    pass