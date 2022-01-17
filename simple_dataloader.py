import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
import matplotlib.pyplot as plt

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


def get_rays(
    height, 
    width, 
    focal, 
    pose,
):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays.
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )
    # i :
    #[[0, 1, 2, ..., w-1]
    # [0, 1, 2, ..., w-1]
    #  ...          ...
    # [0, 1, 2, ..., w-1]]
    # j :
    #[[0  , 0  , 0  , ..., 0  ]
    # [1  , 1  , 1  , ..., 1  ]
    #  ...          ...
    # [h-1, h-1, h-1, ..., h-1]]

    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # 정규 이미지 평면으로 올리기
    
    # Normalize the x axis coordinates.
    # Normalize the y axis coordinates.
    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)
    # directions.shape : (w, h, 3), 각각이 담고 있는 것은 영상의 정규화좌표계의 homogeneous coordiante
    # e.g. (x=10, y=20) 인 곳의 directions 행렬 - [normalized_plane(10), norm(20), 1]
    # NOTE: (h, w, 2) 이 아니라 (w, h, 3) 임에 주의

    # Get the camera matrix.
    rotation = pose[:3, :3]
    translation = pose[:3, -1] 

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :] # (w, h, 1, 3)
    camera_dirs = tf.math.multiply(transformed_dirs, rotation)
    #camera_dirs = transformed_dirs * rotation # 물리공간(월드) 언어(좌표계)로, 정규화이미지평면 위 존재하는 좌표를 번역하려고 함.
    # 여기서 일어나는 연산 : (1, 3) -> [[norm(x), -norm(y), -1]] -> (3, 3) broadcasting 
    #[[norm(x)*R_x_x, -norm(y)*R_y_x, -1*R_z_x],
    # [norm(x)*R_x_y, -norm(y)*R_y_y, -1*R_z_y],
    # [norm(x)*R_x_z, -norm(y)*R_y_z, -1*R_z_z]]
    # 이것이 영상의 모든 픽셀 (모든 x, y) 에 대해서 적용된다고 생각하면 됨.
    
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1) # 물리공간(월드) 언어(좌표계)로, 정규화이미지평면 위 존재하는 점의 좌표값을 얻음.
    #[[norm(x)*R_x_x + -norm(y)*R_y_x + -1*R_z_x], -> [X,
    # [norm(x)*R_x_y + -norm(y)*R_y_y + -1*R_z_y], ->  Y,
    # [norm(x)*R_x_z + -norm(y)*R_y_z + -1*R_z_z]] ->  Z]
    # (w, h, 3)

    ray_origins = tf.broadcast_to(translation, tf.shape(ray_directions)) # (3,) -> (w, h, 3)

    # Return the origins and directions.
    return (ray_origins, ray_directions)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, 3], dtype=tf.float32),  # x
        tf.TensorSpec(shape=[], dtype=tf.int32),           # pos_encoding_dims
    ]
)
def encode_position(
    x,
    pos_encoding_dims
):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    ret = tf.zeros_like(x)

    ret = tf.concat([ret, x], axis=-1)
    for i in tf.range(pos_encoding_dims):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(ret, tf.TensorShape([None, None]))]
        )
        i = tf.cast(i, tf.float32)
        ret = tf.concat([ret, tf.sin(2.0 ** i * x)], axis=-1)
        ret = tf.concat([ret, tf.cos(2.0 ** i * x)], axis=-1)
    return ret[:,3:]

"""
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),  # ray_origins
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),  # ray_directions
        tf.TensorSpec(shape=[], dtype=tf.float32),        # near
        tf.TensorSpec(shape=[], dtype=tf.float32),        # far
        tf.TensorSpec(shape=[], dtype=tf.int32),        # n_samples_per_ray
        tf.TensorSpec(shape=[], dtype=tf.bool),         # rand
    ]
)"""
def render_flat_rays(
    ray_origins, 
    ray_directions, 
    near, 
    far, 
    n_samples_per_ray, 
    rand,
):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        n_samples_per_ray: Number of sample points in a ray.
        rand: Choice for randomising the sampling strategy.

    Returns:
        Tuple of flattened rays and sample points on each rays.
    """
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = tf.linspace(near, far, n_samples_per_ray) # (n,)
    t_vals = tf.cast(t_vals, tf.float32)
    # e.g. (2., 2.1, 2.2, ... , 6)
    if rand:
        # Inject uniform noise into sample space to make the sampling
        # continuous
        shape = tf.Variable(
            initial_value=[0, 0, 0,], 
            trainable=False, 
            dtype=tf.int32
        )
        shape[:-1].assign(ray_origins.shape[:-1])
        shape[ -1].assign(n_samples_per_ray)
        noise = tf.random.uniform(shape=shape) # generate 0~1 uniform nosie
        noise = noise * tf.cast(far-near, tf.float32) # generate 0~(far-near) uniform noise
        noise = noise / tf.cast(n_samples_per_ray, np.float32) # generate 0~((far-near)/n) uniform noise
        # noise : (n,)
        t_vals = t_vals + noise # (n) + (h, w, n) -> (h, w, n)

    # Equation: r(t) = o + td -> Building the "r" here.
    o = ray_origins[..., None, :]
    td = ray_directions[..., None, :] * t_vals[..., None]
    rays = tf.cast(o, dtype=tf.float32) + td
    # if cond : (h, w, 1, 3) + ((h, w, 1, 3) * (n, 1,))
    # else    : (h, w, 1, 3) + ((h, w, 1, 3) * (n, 1,))
    # broadcasting (n, 1) -> (n, 3), (h, w, 1, 3) -> (h, w, n, 3)
    # res     : (h, w, n, 3) = (h, w, 1, 3) + (h, w, n, 3)
    # 해석: 월드좌표계의 언어로 해석한, 정규이미지평면 위의 점들인 (w, h) 격자 각각에 n 개씩 존재함.
    rays_flat = tf.reshape(rays, [-1, 3])
    return (rays_flat, t_vals)

"""
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[4, 4], dtype=tf.float32),  # pose
        tf.TensorSpec(shape=[], dtype=tf.int32),        # image_h
        tf.TensorSpec(shape=[], dtype=tf.int32),        # image_w
        tf.TensorSpec(shape=[], dtype=tf.float32),      # focal_length
        tf.TensorSpec(shape=[], dtype=tf.int32),        # n_samples_per_ray
        tf.TensorSpec(shape=[], dtype=tf.int32),        # pos_encoding_dims
    ]
)"""
def pose_to_ray(
    pose,
    image_h,
    image_w,
    focal_length,
    n_samples_per_ray,
    pos_encoding_dims,
):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: A pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the camera pose.
    """
    (ray_origins, ray_directions) = get_rays(
        height=image_h,
        width=image_w,
        focal=focal_length,
        pose=pose,
    )
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        n_samples_per_ray=n_samples_per_ray,
        rand=True,
    )
    rays_flat = encode_position(
        rays_flat, 
        pos_encoding_dims=pos_encoding_dims)
    return (rays_flat, t_vals)


def map_fn(pose, nerf_params):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: A pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(
        height=nerf_params.image_h, 
        width=nerf_params.image_w, 
        focal=nerf_params.focal_length,
        pose=pose
    )
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        n_samples_per_ray=nerf_params.n_samples_per_ray,
        rand=True,
    )
    return (rays_flat, t_vals)


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
    # Load numpy formed data.
    images, poses, focal_length = get_np_data_from_local_file('./tiny_nerf_data.npz')
    num_images, image_h, image_w, _ = images.shape
    
    # Save params.
    nerf_params = NeRFParams(
        image_h=image_h,
        image_w=image_w,
        focal_length=focal_length,
    )

    # Get dataset.
    train_ds, val_ds = get_train_val_tf_ds(
        images, 
        poses, 
        nerf_params
    )

    # Test.
    it = iter(train_ds)
    next(it)
