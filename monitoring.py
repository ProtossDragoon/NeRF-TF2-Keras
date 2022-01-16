# Internal Packages and Modules
from datetime import datetime

# Third party Packages and Modules
import tensorflow as tf
from tensorflow.python.util.compat import path_to_str
import matplotlib.pyplot as plt
import numpy as np
import imageio
import tqdm
import json

# NeRF project
from utils import runtime
from parameters import NeRFParams
from nerf_core.nerf import NeRFModel


class SaveBestModel(tf.keras.callbacks.ModelCheckpoint):
    
    def __init__(
        self,
        weights_save_dir,
        nerf_params,
        mode = "min",
        monitor = "val_loss"
    ):
        self.weights_save_dir = weights_save_dir
        self.nerf_params = nerf_params
        self.mode = mode
        self.monitor = monitor        
        super().__init__(
            self.get_new_savedmodel_dir(),
            monitor=monitor,
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            save_freq="epoch",
        )

    def get_new_savedmodel_dir(self):
        now = f'{datetime.now().strftime("%Y%m%d_%H:%M:%S")}'
        new_dir = tf.io.gfile.join(
            self.weights_save_dir,
            f'{now}_{self.mode}_{self.monitor}',
        )
        return path_to_str(new_dir)
    
    def set_new_savedmodel_dir(self):
        #NOTE: self.filepath : superclass' member variable.
        self.savedmodel_dir = self.filepath = self.get_new_savedmodel_dir()

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):
        self.set_new_savedmodel_dir()
        super().on_epoch_end(epoch, logs)
        if tf.io.gfile.exists(self.savedmodel_dir):
            path = tf.io.gfile.join(self.savedmodel_dir, 'nerf_params.json')
            with open(path, 'w') as f:
                json.dump(self.nerf_params.__dict__, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TrainingMonitor(tf.keras.callbacks.Callback):
    
    def __init__(
        self,
        sample_batched_test_rays_flat,
        sample_batched_test_t_vals,
        nerf_model:NeRFModel,
        nerf_params:NeRFParams,
        close_fig:bool=True,
        save_dir:str='./result_images',
    ):
        assert sample_batched_test_rays_flat.shape[0] == nerf_params.BATCH_SIZE
        assert sample_batched_test_t_vals.shape[0] == nerf_params.BATCH_SIZE

        super().__init__()
        self.sample_batched_test_rays_flat = sample_batched_test_rays_flat
        self.sample_batched_test_t_vals = sample_batched_test_t_vals
        self.nerf_params = nerf_params
        self.nerf_model = nerf_model
        self.loss_list = []
        self.close_fig = close_fig
        self.save_dir = save_dir
        # Create a directory to save the images during training.
        tf.io.gfile.makedirs(save_dir)
    
    def on_train_end(
        self,
        logs=None,
    ):
        def create_gif(path_to_images, giffile_name):
            filenames = tf.io.gfile.glob(path_to_images)
            filenames = sorted(filenames)
            images = []
            for filename in tqdm.tqdm(filenames):
                images.append(imageio.imread(filename))
            kwargs = {"duration": 0.25}
            imageio.mimsave(giffile_name, images, "GIF", **kwargs)

        images_path = tf.io.gfile.join(self.save_dir, '*.png')
        giffile_name = tf.io.gfile.join(self.save_dir, 'result.gif')
        create_gif(images_path, giffile_name)
        
    def on_epoch_end(
        self, 
        epoch,
        logs=None,
    ):
        test_recons_images, depth_maps = self.nerf_model.render_rgb_depth(
            rays_flat=self.sample_batched_test_rays_flat,
            t_vals=self.sample_batched_test_t_vals,
        )

        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(tf.keras.preprocessing.image.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(tf.keras.preprocessing.image.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        self.loss_list.append(logs["loss"])
        ax[2].plot(self.loss_list)
        ax[2].set_xticks(np.arange(0, self.nerf_params.EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(tf.io.gfile.join(self.save_dir, f'{epoch:03d}.png'))

        if not runtime.RuntimeChecker.colab_mode:
            plt.show(block=(not self.close_fig))
            if self.close_fig:
                plt.pause(1)
            plt.close()