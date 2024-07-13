import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np
import keras_nlp
from keras import ops
import include.loss as loss

# Model params.
NUM_LAYERS = 3
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5


@keras.saving.register_keras_serializable()
class EmissionPredictor(keras.Model):
    def __init__(self, image_size, embedding_layer, bottom_layers, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.embedding_layer.patch_encoder.downstream = True
        self.bottom_layers = bottom_layers
        self.image_size = image_size

    def build(self, input_shape):
        self.flatten = keras.layers.Flatten()
        self.predictor = keras.Sequential([
            keras.layers.Dense(INTERMEDIATE_DIM, activation='relu'),
            keras.layers.Dense(1)
        ])
        patch_layer = self.embedding_layer.patch_layer
        patch_encoder = self.embedding_layer.patch_encoder
        encoder = self.embedding_layer.encoder
        inputs = keras.Input(shape=input_shape[1:])
        x = keras.layers.Resizing(
            self.image_size, self.image_size)(inputs)

        x = patch_layer(inputs)
        x = patch_encoder(x)
        x = encoder(x)

        embedding_shape = x.shape
        inputs = keras.Input(
            shape=(input_shape[0], embedding_shape[1]*embedding_shape[2]))

        self.emiss_trans = EmissTransformer(
            embedding_shape[1]*embedding_shape[2])
        outputs = self.emiss_trans(inputs)
        self.predictor(outputs)

    def call(self, inputs):
        x = inputs
        if self.bottom_layers is not None:
            x = self.bottom_layers(inputs)
        x = self.embedding(x)
        outputs = self.emiss_trans(x)
        return self.predictor(outputs)

    def calculate_loss(self, inputs):
        x, y1, y2 = inputs[0], inputs[1], inputs[2]
        if self.bottom_layers is not None:
            x = self.bottom_layers(x)

        o1 = self.embedding(x)
        o1 = self.emiss_trans(o1)
        o1_y = self.do_embedding(y1)
        loss1 = keras.losses.MeanSquaredError()(o1_y, o1)

        o2 = self.predictor(o1)
        loss2 = keras.losses.MeanAbsoluteError()(y2, o2)
        return 0.8*loss1+0.2*loss2, y2, o2

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, loss_y, loss_pred = self.calculate_loss(inputs)

         # Apply gradients.
        train_vars = [
            self.predictor.trainable_variables,
            self.emiss_trans.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_y, loss_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        total_loss, loss_y, loss_pred = self.calculate_loss(inputs)
        # Update the trackers.
        self.compiled_metrics.update_state(loss_y, loss_pred)
        return {m.name: m.result() for m in self.metrics}

    def do_embedding(self, input):
        outputs = keras.layers.Resizing(
            self.image_size, self.image_size)(input)
        patch_layer = self.embedding_layer.patch_layer
        patch_encoder = self.embedding_layer.patch_encoder
        encoder = self.embedding_layer.encoder

        patches = patch_layer(outputs)
        unmasked_embeddings = patch_encoder(patches)
        # Pass the unmaksed patch to the encoder.
        embedding = encoder(unmasked_embeddings)
        return self.flatten(embedding)

    def embedding(self, inputs):
        embedding = tf.map_fn(lambda x: self.do_embedding(x), inputs)
        positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
        return embedding + positional_encoding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_layer": keras.saving.serialize_keras_object(self.embedding_layer),
                "bottom_layers": keras.saving.serialize_keras_object(self.bottom_layers),
                "image_size": self.image_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        for k in ["embedding_layer", "bottom_layers"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)


@keras.saving.register_keras_serializable()
class EmissTransformer(keras.Model):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.transformer_encoder = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            layer_norm_epsilon=NORM_EPSILON,
        )
        self.dense = keras.layers.Dense(self.embed_dim)
        self.dropout = keras.layers.Dropout(DROPOUT)
        self.layernorm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        mask = compute_mask(batch_size, seq_len, seq_len, "bool")
        out1 = inputs
        for i in range(NUM_LAYERS):
            out1 = self.transformer_encoder(
                out1, attention_mask=mask)
        out1 = self.flatten(out1)
        out2 = self.dense(out1)
        out2 = self.dropout(out2)
        return self.layernorm(out2)


def compute_mask(batch_size, n_dest, n_src, dtype):
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


def emission_predictor(input_shape, image_size, embedding, bottom_layers):
    predictor = EmissionPredictor(image_size, embedding, bottom_layers)
    predictor.build(input_shape)
    return predictor
