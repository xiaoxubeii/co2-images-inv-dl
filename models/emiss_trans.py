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
    def __init__(self, image_size, mae, bottom_layers, **kwargs):
        super().__init__(**kwargs)
        self.mae = mae
        self.mae.patch_encoder.downstream = True
        self.trans = EmissTransformer()
        self.predictor = keras.Sequential([
            keras.layers.Dense(1, activation='linear'),
            keras.layers.Dense(1, activation='relu')
        ])
        self.bottom_layers = bottom_layers
        self.image_size = image_size

    def call(self, inputs):
        if self.bottom_layers is not None:
            outputs = self.bottom_layers(inputs)
        outputs = self.embedding(outputs)
        outputs = self.trans(outputs)
        return self.predictor(outputs)

    def calculate_loss(self, inputs):
        x, y = inputs[0], inputs[1]
        if self.bottom_layers is not None:
            x = self.bottom_layers(x)

        o1 = self.embedding(x)
        y1 = self.trans(o1)
        loss1 = keras.losses.MeanSquaredError()(y1, o1)
        o2 = self.predictor(y1)
        loss2 = keras.losses.MeanAbsoluteError()(y, o2)
        return 20*loss1+loss2, y, o2

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, loss_y, loss_pred = self.calculate_loss(inputs)

         # Apply gradients.
        train_vars = [
            self.predictor.trainable_variables,
            self.trans.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_pred, loss_y)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        total_loss, loss_y, loss_pred = self.calculate_loss(inputs)
        # Update the trackers.
        self.compiled_metrics.update_state(loss_y, loss_pred)
        return {m.name: m.result() for m in self.metrics}

    def embedding(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        def _embedding(inputs):
            outputs = keras.layers.Resizing(
                self.image_size, self.image_size)(inputs)
            patch_layer = self.mae.patch_layer
            patch_encoder = self.mae.patch_encoder
            encoder = self.mae.encoder

            patches = patch_layer(outputs)
            unmasked_embeddings = patch_encoder(patches)
            # Pass the unmaksed patch to the encoder.
            return encoder(unmasked_embeddings)

        embedding = tf.map_fn(_embedding, inputs)
        positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
        embedding = embedding + positional_encoding
        embedding_shape = embedding.shape
        return tf.reshape(
            embedding, [batch_size, seq_len, embedding_shape[-1]*embedding_shape[-2]])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mae": self.mae,
                "bottom_layers": self.bottom_layers,
                "image_size": self.image_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        for k in ["mae"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)


@keras.saving.register_keras_serializable()
class EmissTransformer(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.mae = mae
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=NORM_EPSILON)
        self.dropout = keras.layers.Dropout(rate=DROPOUT)
        self.transformer_encoder = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            layer_norm_epsilon=NORM_EPSILON,
        )

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        mask = compute_mask(batch_size, seq_len, seq_len, "bool")
        # embedding = self.embedding(inputs)
        # Apply layer normalization and dropout to the embedding.
        # outputs = self.layer_norm(inputs)
        # outputs = self.dropout(outputs)

        # Add a number of encoder blocks
        outputs = inputs
        for i in range(NUM_LAYERS):
            outputs = self.transformer_encoder(outputs, attention_mask=mask)
        return outputs

    # def calculate_loss(self, inputs):
    #     return self.loss_func(embedding, outputs), outputs


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


def emission_predictor(input_shape, image_size, autoencoder, bottom_layers=None):
    inputs = keras.Input(shape=input_shape)
    predictor = EmissionPredictor(
        image_size,  autoencoder, bottom_layers=bottom_layers)
    predictor(inputs)
    return predictor
