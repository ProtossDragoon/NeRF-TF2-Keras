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
    (ray_origin, ray_directions) = get_rays(
        height=image_h,
        width=image_w,
        focal=focal_length,
        pose=pose,
    )
    (rays_flat, ray_t) = render_flat_rays(
        ray_origin=ray_origin,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        n_samples_per_ray=n_samples_per_ray,
        rand=True,
    )
    rays_flat = encode_position(
        rays_flat, 
        pos_encoding_dims=pos_encoding_dims)
    return (rays_flat, ray_t)


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
    pixel_directions = tf.stack([transformed_x, -transformed_y, -tf.ones_like(mesh_x)], axis=-1)

    # Get the camera extrinsic matrix.
    camera_rotation = pose[:3, :3]
    camera_translation = pose[:3, -1]

    # Get origin and directions for the rays.    
    ## directions
    pixel_directions = pixel_directions[..., None, :] # (w, h, 1, 3)
    ray_directions = tf.math.multiply(pixel_directions, camera_rotation) # (w, h, 3, 3)
    ray_directions = tf.reduce_sum(pixel_directions, axis=-1) # (w, h, 3)
    ## origin
    ray_origin = tf.broadcast_to(camera_translation, tf.shape(ray_directions)) # (3,) -> (w, h, 3)

    # Return the origin and directions.
    return (ray_origin, ray_directions)


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
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),  # ray_origin
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),  # ray_directions
        tf.TensorSpec(shape=[], dtype=tf.float32),        # near
        tf.TensorSpec(shape=[], dtype=tf.float32),        # far
        tf.TensorSpec(shape=[], dtype=tf.int32),        # n_samples_per_ray
        tf.TensorSpec(shape=[], dtype=tf.bool),         # rand
    ]
)"""
def render_flat_rays(
    ray_origin, 
    ray_directions, 
    near, 
    far, 
    n_samples_per_ray, 
    rand,
):
    """Renders the rays and flattens it.

    Args:
        ray_origin: The origin points for rays.
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
    ray_t = tf.linspace(near, far, n_samples_per_ray) # (n,)
    ray_t = tf.cast(ray_t, tf.float32)
    # e.g. (2., 2.1, 2.2, ... , 6)
    
    if rand:
        # Inject uniform noise into sample space to make the sampling continuous.
        shape = tf.Variable(
            initial_value=[0, 0, 0,], 
            trainable=False, 
            dtype=tf.int32
        )
        shape[:-1].assign(ray_origin.shape[:-1])
        shape[ -1].assign(n_samples_per_ray)
        noise = tf.random.uniform(shape=shape) # generate 0~1 uniform nosie
        noise = noise * tf.cast(far-near, tf.float32) # generate 0~(far-near) uniform noise
        noise = noise / tf.cast(n_samples_per_ray, tf.float32) # generate 0~((far-near)/n) uniform noise
        ray_t = ray_t + noise # (n) + (h, w, n) -> (h, w, n)
    
    # if condition
    ## ray_t.shape: (h, w, n)
    # else condition
    ## ray_t.shape: (n,)

    # Equation: r(t) = o + td -> Building the "r" here.
    o = ray_origin[..., None, :]
    td = ray_directions[..., None, :] * ray_t[..., None]
    rays = tf.cast(o, dtype=tf.float32) + td
    
    # if condition
    ## o.shape: (h, w, 1, 3)
    ## td.shape: (h, w, n, 3) = ((h, w, 1, 3) * (h, w, n, 1))
    ## rays.shape: (h, w, n, 3)
    # else condition
    ## o.shape: (h, w, 1, 3)
    ## td.shape: (h, w, n, 3) = ((h, w, 1, 3) * (n, 1))
    ## rays.shape: (h, w, n, 3)

    rays_flat = tf.reshape(rays, [-1, 3])

    # Always
    ## rays_flat.shape: (h*w*n, 3)

    return (rays_flat, ray_t)


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