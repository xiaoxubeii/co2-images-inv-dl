import keras


def co2emiss_regres(input_shape, embedding, **kwargs):
    inputs = keras.Input(shape=input_shape, batch_size=32)
    x = embedding.patch_layer(inputs)
    x = embedding.patch_encoder(x)
    x = embedding.encoder(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.LeakyReLU(alpha=0.3)(x)
    return keras.Model(inputs, outputs, name="co2emiss_regres")


# @keras.saving.register_keras_serializable()
# class EmissRegression(keras.Model):
#     def __init__(self, embedding_layer, bottom_layers=None, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_layer = embedding_layer
#         self.bottom_layers = bottom_layers

#     def build(self, input_shape):
#         inputs = keras.Input(shape=input_shape)
#         if self.bottom_layers is not None:
#             inputs = self.bottom_layers(inputs)
#         x = self.embedding_layer(inputs)
#         if isinstance(x, list):
#             x = x[0]
#         self.quantifying = keras.Sequential([
#             keras.layers.Dense(128, activation='relu',
#                                input_shape=x.shape[1:]),
#             keras.layers.Dense(1),
#             keras.layers.LeakyReLU(alpha=0.3)
#         ])
#         self.quantifying(x)

#     def call(self, inputs):
#         if self.bottom_layers is not None:
#             inputs = self.bottom_layers(inputs)
#         x = self.embedding_layer(inputs)
#         return self.quantifying(x)

#     @ classmethod
#     def from_config(cls, config):
#         for k in ["embedding_layer"]:
#             config[k] = keras.saving.deserialize_keras_object(config[k])
#         return cls(**config)

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "embedding_layer": keras.saving.serialize_keras_object(self.embedding_layer),
#             }
#         )
#         return config
