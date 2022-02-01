# Internal Packages and Modules
import tensorflow as tf

# NeRFModel project
from nerf_core import architecture
from parameters import NeRFParams


class ValidateNeRFModel(tf.keras.Model):
    # TODO

    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)


class NeRFModel(ValidateNeRFModel):
    
    def __init__(
        self, 
        nerf_architecture:architecture.NeRFArchitecture,
        nerf_params:NeRFParams,
    ):
        super().__init__(name=f'nerf_{nerf_architecture.name}')
        self.nerf_architecture = nerf_architecture
        self.nerf_params = nerf_params

    def compile(
        self, 
        optimizer, 
        loss_fn,
    ):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="psnr")

    def call(
        self,
        rays,
    ):
        rays_flat, ray_t = rays
        return self.render_rgb_depth(
            rays_flat=rays_flat,
            ray_t=ray_t,
        )

    def train_step(
        self, 
        inputs
    ):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, ray_t) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model.
            rgb, _ = self.render_rgb_depth(
                rays_flat=rays_flat, 
                ray_t=ray_t, 
            )
            loss = self.loss_fn(images, rgb)

        # Get the trainable variables.
        trainable_variables = self.nerf_architecture.trainable_variables

        # Get the gradeints of the trainiable variables with respect to the loss.
        gradients = tape.gradient(loss, trainable_variables)

        # Apply the grads and optimize the model.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(
        self, 
        inputs
    ):
        # Get the images and the rays.
        (images, rays) = inputs
        # Tuple of flattened rays and sample points corresponding to the camera pose.
        (rays_flat, ray_t) = rays

        # Get the predictions from the model.
        rgb, _ = self.render_rgb_depth( 
            rays_flat=rays_flat, 
            ray_t=ray_t, 
        )
        loss = self.loss_fn(images, rgb)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)

        # NOTE: Prefix the name with "val_" to monitor validation metrics.
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]

    def render_rgb_depth(
        self,
        rays_flat, 
        ray_t,
        rand=True,
    ):
        """Generates the RGB image and depth map from model prediction.

        Args:
            rays_flat: The flattened rays 
                that serve as the input to the NeRF model.
            ray_t: The sample points for the rays.
            rand: Choice to randomise the sampling strategy.

        Returns:
            Tuple of rgb image and depth map.
        """
        # Get the predictions from the nerf model and reshape it.
        predictions = self.nerf_architecture(rays_flat)
        predictions = tf.reshape(predictions, shape=(
                self.nerf_params.batch_size, 
                self.nerf_params.image_h,
                self.nerf_params.image_w,
                self.nerf_params.n_samples_per_ray, 
                4 # output = tf.kears.layers.Dense(units=4)(x)
                )
            )

        # Slice the predictions into rgb and sigma.
        rgb = tf.sigmoid(predictions[..., :-1])
        sigma_a = tf.nn.relu(predictions[..., -1])

        # Get the distance of adjacent intervals.
        delta = ray_t[..., 1:] - ray_t[..., :-1]
        if rand:
            delta = tf.concat(
                [delta, tf.broadcast_to([1e10], shape=(self.nerf_params.batch_size, self.nerf_params.image_h, self.nerf_params.image_w, 1))], 
                axis=-1
            )
            alpha = 1.0 - tf.exp(-sigma_a * delta)
        else:
            delta = tf.concat(
                [delta, tf.broadcast_to([1e10], shape=(self.nerf_params.batch_size, 1))], 
                axis=-1
            )
            alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

        # Get transmittance.
        e = 1e-10
        transmittance = tf.math.cumprod((1.0-alpha)+e, axis=-1, exclusive=True)
        weights = alpha * transmittance
        rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

        # Get depth
        if rand:
            depth_map = tf.reduce_sum(weights * ray_t, axis=-1)
        else:
            depth_map = tf.reduce_sum(weights * ray_t[:, None, None], axis=-1)

        return (rgb, depth_map)