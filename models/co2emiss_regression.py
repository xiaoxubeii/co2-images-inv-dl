import keras


def co2emiss_regres(input_shape, embedding, bottom_layers, **kwargs):
    pretrained_embedding_layer = keras.Sequential([
        embedding.patch_layer,
        embedding.patch_encoder,
        embedding.encoder,
        keras.layers.Flatten(),
    ])

    regres = EmissRegression(pretrained_embedding_layer, bottom_layers)
    regres.build(input_shape)
    return regres


class EmissRegression(keras.Model):
    def __init__(self, embedding_layer, bottom_layers, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.quanifying = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1),
            keras.layers.LeakyReLU(alpha=0.3)
        ])
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
