# Third party Packages and Modules
import tensorflow as tf


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
    mesh_x, mesh_y = tf.meshgrid(
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
    transformed_x = image_plane_to_normalized_plane(mesh_x, focal, width * 0.5,)
    transformed_y = image_plane_to_normalized_plane(mesh_y, focal, height * 0.5,)

    # Create the direction unit vectors.
    # directions.shape : (w, h, 3), homogeneous coordiante
    directions = tf.stack([transformed_x, -transformed_y, -tf.ones_like(mesh_x)], axis=-1)

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


def image_plane_to_normalized_plane(
    meshgrid, f, c,
):
    """
    Args:
        meshgrid [Tensor]: A axis' meshgrid of pixel coordinate (image plane coord).
            - Tensor shape should be [h, w].
            - Tensor element value range should be [0, w-1] or [0, h-1].
        f: A axis value of focal length of the camera.
        c: A axis value of principal point of the camera.
    
    Returns:
        meshgrid [Tensor]: Normalized meshgrid of the axis.
            Normalized meshgrid is on the normalized image plane.
            Tensor shape would be same as input Tensor.
    """
    return (meshgrid - c) / f


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
        noise = noise / tf.cast(n_samples_per_ray, tf.float32) # generate 0~((far-near)/n) uniform noise
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


if __name__ == '__main__':
    pass