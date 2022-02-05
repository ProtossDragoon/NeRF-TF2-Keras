# Third party Packages and Modules
import tensorflow as tf

# NeRF project
import parameters


class ValidateNeRFArchitecture():
    # TODO
    pass


class NeRFArchitecture(ValidateNeRFArchitecture):
    def __init__(
        self,
        nerf_params,
        n_layers:int=8,
    ):
        """Generates the NeRFModel neural network.

        Args:
            n_layers: The number of MLP layers.
            nerf_params: The parameter server.

        Returns:
            The [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) model.
        """
        self.n_layers = n_layers
        self.nerf_params = nerf_params


class DNNArchitecture(NeRFArchitecture):
    def __init__(
        self,
        nerf_params:parameters.NeRFParams,
        n_layers:int=8,
        architecture_name ='dnn'
    ):
        super().__init__(
            n_layers=n_layers,
            nerf_params=nerf_params,
        )
        self.architecture_name = architecture_name

    def get_nerf_architecture(
        self,
    )->tf.keras.Model:
        num = self.nerf_params.get_number_of_flatten_rays()
        dim = self.nerf_params.get_dimension_of_network_input_ray()
        # Input layer
        input = tf.keras.Input(shape=(num, dim))
        x = input
        # Backbone
        for i in range(self.n_layers):
            x = tf.keras.layers.Dense(units=64, activation="relu")(x)
            if i % 4 == 0 and i > 0:
                # Inject residual connection.
                x = tf.keras.layers.concatenate([x, input], axis=-1)
        # Output layer
        output = tf.keras.layers.Dense(units=4)(x)
        # Wrap as keras Model
        model =  tf.keras.Model(
            inputs=input, 
            outputs=output, 
            name=self.architecture_name
        )
        model.compile(loss='') # NOTE: Model should be compiled even if it is just backbone architecture. Use Layer instead.
        return model