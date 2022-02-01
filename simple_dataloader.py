# Third party Packages and Modules
import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
import matplotlib.pyplot as plt

# NeRFModel project
from nerf_core.ray import pose_to_ray
from parameters import NeRFParams


def get_np_data_from_local_file(file_name, visualize_sample=False):
    if tf.io.gfile.exists(file_name):
        data = np.load(file_name)
    else:
        raise FileNotFoundError(f'File {file_name} not found!')

    images = data['images']
    poses = data['poses']
    focal_length = data['focal']

    if visualize_sample:
        # Plot a random image from the dataset for visualization.
        plt.imshow(images[np.random.randint(low=0, high=images.shape[0])])
        plt.show()

    return images, poses, focal_length


def get_zip_from_ds(np_images, np_poses, nerf_params):
    # The dataset contains (image, ray)
    _image_ds = tf.data.Dataset.from_tensor_slices(np_images)
    _pose_ds = tf.data.Dataset.from_tensor_slices(np_poses)
    # FIXME: dataset.map() 은 lambda 로 사용할 수 없다! 하려면 py_function 을 써야 함.
    # NOTE: py_function 쓰는 순간 TPU 못건드리는거 알지?
    _ray_ds = _pose_ds.map(lambda pose:
        pose_to_ray(
            pose,
            # to use TPU
            tf.constant(nerf_params.image_h, dtype=tf.int32),
            tf.constant(nerf_params.image_w, dtype=tf.int32),
            tf.constant(nerf_params.focal_length, dtype=tf.float32),
            tf.constant(nerf_params.n_samples_per_ray, dtype=tf.int32),
            tf.constant(nerf_params.pos_encoding_dims, dtype=tf.int32),
            ),
        num_parallel_calls=nerf_params.AUTO)
    zip_ds = tf.data.Dataset.zip((_image_ds, _ray_ds))
    return zip_ds


def get_train_ds(train_images, train_poses, nerf_params):
    # Make the training pipeline.
    _train_zip_ds = get_zip_from_ds(train_images, train_poses, nerf_params)
    train_ds = (
        _train_zip_ds
        .shuffle(nerf_params.batch_size)
        .batch(nerf_params.batch_size, drop_remainder=True, num_parallel_calls=nerf_params.AUTO)
        .prefetch(nerf_params.AUTO)
    )
    return train_ds


def get_val_ds(val_images, val_poses, nerf_params):
    # Make the validation pipeline.
    _val_zip_ds = get_zip_from_ds(val_images, val_poses, nerf_params)
    val_ds = (
        _val_zip_ds
        .shuffle(nerf_params.batch_size)
        .batch(nerf_params.batch_size, drop_remainder=True, num_parallel_calls=nerf_params.AUTO)
        .prefetch(nerf_params.AUTO)
    )
    return val_ds


def get_train_val_tf_ds(
    images, 
    poses, 
    nerf_params,
):
    # image, ray 로 구성된 tensorflow dataset
    # training, validation 둘 모두를 리턴합니다.

    # Create the training split.
    num_images = images.shape[0]
    split_index = int(num_images * nerf_params.TRAIN_TEST_SPLIT)

    # Split the images into training and validation.
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Split the poses into training and validation.
    train_poses = poses[:split_index]
    val_poses = poses[split_index:]

    # get tensorflow dataset
    train_ds = get_train_ds(train_images, train_poses, nerf_params)
    val_ds = get_val_ds(val_images, val_poses, nerf_params)

    return train_ds, val_ds


if __name__ == '__main__':
    pass