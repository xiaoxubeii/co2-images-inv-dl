import keras


def co2emiss_regres(input_shape, embedding, bottom_layers, **kwargs):
    pretrained_embedding_layer = keras.Sequential([
        embedding.patch_layer,
        embedding.patch_encoder,
        embedding.encoder,
        keras.layers.Flatten(),
    ], name="embedding")

    regres = EmissRegression(pretrained_embedding_layer, bottom_layers)
    regres.build(input_shape)
    return regres


@keras.saving.register_keras_serializable()
class EmissRegression(keras.Model):
    def __init__(self, embedding_layer, bottom_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.quanifying = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1),
            keras.layers.LeakyReLU(alpha=0.3)
        ], name="regressor")
        self.bottom_layers = bottom_layers

    def build(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.embedding_layer(inputs)
        self.quanifying(x)

    def save_model(self):
        return self.quanifying

    def call(self, inputs):
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.embedding_layer(inputs)
        return self.quanifying(x)

    @classmethod
    def from_config(cls, config):
        for k in ["embedding_layer"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_layer": keras.saving.serialize_keras_object(self.embedding_layer),
            }
        )
        return config
