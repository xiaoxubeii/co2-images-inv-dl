import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import numpy as np
import keras_nlp
from keras import ops

# Model params.
NUM_LAYERS = 3
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5


class EmissPredictor(keras.Model):
    def __init__(self, mae, bottom_layers, top_layers, **kwargs):
        super().__init__(**kwargs)
        self.mae = mae
        self.trans = EmissTransformer(mae)
        self.predictor = keras.layers.Sequential([
            keras.layers.Dense(1, activation='linear'),
            keras.layers.Dense(1, activation='relu')
        ])
        self.bottom_layers = bottom_layers
        self.top_layers = top_layers

    def calculate_loss(self, inputs, test=False):
        x, y = inputs
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)

        loss1, outputs = self.trans.calculate_loss(x)
        outputs = self.predictor(outputs)
        loss2 = self.compiled_loss(y, outputs)
        return loss1+loss2

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(inputs)

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
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        total_loss, loss_patch, loss_output = self.calculate_loss(
            inputs, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}


class EmissTransformer(keras.Model):
    def __init__(self, mae, **kwargs):
        super().__init__(**kwargs)
        self.mae = mae

    def calculate_loss(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        mask = compute_mask(batch_size, seq_len, seq_len, "bool")

        def do_embedding(inputs):
            patch_layer = self.mae.patch_layer
            patch_encoder = self.mae.patch_encoder
            patch_encoder.downstream = True
            encoder = self.mae.encoder

            patches = patch_layer(inputs)
            unmasked_embeddings = patch_encoder(patches)
            # Pass the unmaksed patch to the encoder.
            return encoder(unmasked_embeddings)

        embedding = tf.vectorized_map(do_embedding, inputs)
        positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
        embedding = embedding + positional_encoding
        embedding_shape = embedding.shape
        embedding = tf.reshape(
            embedding, [batch_size, seq_len, embedding_shape[-1]*embedding_shape[-2]])

        # Apply layer normalization and dropout to the embedding.
        outputs = keras.layers.LayerNormalization(
            epsilon=NORM_EPSILON)(embedding)
        outputs = keras.layers.Dropout(rate=DROPOUT)(outputs)

        # Add a number of encoder blocks
        for i in range(NUM_LAYERS):
            outputs = keras_nlp.layers.TransformerEncoder(
                intermediate_dim=INTERMEDIATE_DIM,
                num_heads=NUM_HEADS,
                dropout=DROPOUT,
                layer_norm_epsilon=NORM_EPSILON,
            )(outputs, attention_mask=mask)

        return self.compiled_loss(embedding, outputs), outputs


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
