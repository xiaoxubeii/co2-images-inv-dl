import keras


def co2emiss_regres(input_shape, embedding_layer, **kwargs):
    inputs = keras.Input(shape=input_shape)
    patch_layer = embedding_layer.patch_layer
    patch_encoder = embedding_layer.patch_encoder
    encoder = embedding_layer.encoder

    patches = patch_layer(inputs)
    unmasked_embeddings = patch_encoder(patches)
    # Pass the unmaksed patch to the encoder.
    embedding = encoder(unmasked_embeddings)
    x = keras.layers.Flatten()(embedding)
    outputs = keras.layers.Dense(1, activation='linear')(x)
    return keras.Model(inputs, outputs)
