# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf

import keras
import numpy as np
import tensorflow as tf

from models.my_efficientnet import EfficientNet
from models.my_essential_inversors import (essential_regressor,
                                           essential_regressor_2,
                                           essential_regressor_3,
                                           linear_regressor)
from models.my_mobilenet import MobileNet
from models.my_shufflenet import ShuffleNet
from models.my_squeezenet import SqueezeNet
from models.cnn_lstm import cnn_lstm
from models.xco2_transformer import xco2_transformer
from models.co2emission_transformer import emission_predictor


def get_preprocessing_layers(
    n_layer: tf.keras.layers.Normalization, n_chans: int, noisy_chans: list, window_length: int
):
    """Return preprocessing layers for regression model."""
    def preproc_layers(x):
        chans = [None] * n_chans
        for idx in range(n_chans):
            if noisy_chans[idx]:
                if window_length > 0:
                    chans[idx] = tf.keras.layers.GaussianNoise(
                        stddev=0.7, name=f"noise_{idx}"
                    )(x[:, :, :, :, idx: idx + 1])
                else:
                    chans[idx] = tf.keras.layers.GaussianNoise(
                        stddev=0.7, name=f"noise_{idx}"
                    )(x[:, :, :, idx: idx + 1])
            else:
                if window_length > 0:
                    # layer = tf.keras.layers.Layer()
                    chans[idx] = x[:, :, :, :, idx: idx + 1]
                else:
                    chans[idx] = x[:, :, :, idx: idx + 1]

        concatted = tf.keras.layers.Concatenate()(chans)
        x = n_layer(concatted)
        return x

    return preproc_layers


def get_top_layers(classes: int, choice_top: str = "linear"):
    """Return top layers for regression model."""

    def top_layers(x):
        if choice_top in [
            "efficientnet",
            "squeezenet",
            "nasnet",
            "mobilenet",
            "shufflenet",
        ]:
            x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
            x = tf.keras.layers.Dense(classes, name="regressor")(x)
            outputs = tf.keras.layers.LeakyReLU(
                alpha=0.3, dtype=tf.float32, name="regressor_activ"
            )(x)
        elif choice_top == "linear":
            outputs = tf.keras.layers.Dense(classes, name="regressor")(x)
        elif choice_top.startswith("essential"):
            x = tf.keras.layers.Dense(1)(x)
            outputs = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        elif choice_top == "cnn-lstm":
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        else:
            return x
        return outputs

    return top_layers


def get_core_model(
    name: str,
    input_shape: list,
    classes: int = 1,
    dropout_rate: float = 0.2,
    scaling_coefficient: float = 1,
    top_layers=None,
    bottom_layers=None,
    config=None,
):
    """Get core model for regression model."""
    if name == "efficientnet":
        core_model = EfficientNet(
            scaling_coefficient=0.5,
            input_shape=input_shape,
            classes=classes,
            dropout_rate=dropout_rate,
        )
    elif name == "linear":
        core_model = linear_regressor(input_shape)
    elif name == "essential":
        core_model = essential_regressor(input_shape)
    elif name == "essential_2":
        core_model = essential_regressor_2(input_shape)
    elif name == "essential_3":
        core_model = essential_regressor_3(input_shape)
    elif name == "squeezenet":
        core_model = SqueezeNet(input_shape, dropout_rate, compression=0.4)
    elif name == "mobilenet":
        core_model = MobileNet(input_shape, scaling_coeff=0.4)
    elif name == "shufflenet":
        core_model = ShuffleNet(input_shape, scaling_coefficient=0.75)
    elif name == "cnn-lstm":
        core_model = cnn_lstm(input_shape)
    elif name == "xco2_transformer":
        core_model = xco2_transformer(input_shape=input_shape, image_size=config.model.image_size, channel_size=input_shape[-1], patch_size=config.model.patch_size,
                                      top_layers=top_layers, bottom_layers=bottom_layers)
    elif name == "co2emission_transformer":
        # xco2t = xco2_transformer(input_shape=input_shape, image_size=config.model.image_size, channel_size=input_shape[-1], patch_size=config.model.patch_size,
        #                          top_layers=top_layers, bottom_layers=bottom_layers)
        # xco2t.load_weights(config.model.embedding_path)
        xco2t = keras.saving.load_model(config.model.embedding_path)
        xco2t.freeze_all_layers()
        core_model = emission_predictor(
            input_shape, config.model.image_size, xco2t, bottom_layers=bottom_layers)

    else:
        sys.exit()

    return core_model


@dataclass
class Reg_model_builder:
    """Return appropriate regression model."""

    name: str = "linear"
    input_shape: list = field(default_factory=lambda: [64, 64, 3])
    classes: int = 1
    n_layer: tf.keras.layers.Normalization = tf.keras.layers.Normalization(
        axis=-1)
    noisy_chans: list = field(
        default_factory=lambda: [True, False, False, False, False]
    )
    dropout_rate: float = 0.2
    scaling_coefficient: float = 1
    window_length: int = 0
    config: DictConfig = None

    def get_model(self):
        """Return regression model, keras or locals."""
        bottom_layers = get_preprocessing_layers(
            self.n_layer, self.input_shape[-1], self.noisy_chans, self.window_length
        )
        top_layers = get_top_layers(self.classes, self.name)
        core_model = get_core_model(
            self.name,
            self.input_shape,
            self.classes,
            self.dropout_rate,
            self.scaling_coefficient,
            top_layers,
            bottom_layers,
            self.config
        )
        if self.name not in ("xco2_transformer", "co2emission_transformer"):
            inputs = tf.keras.layers.Input(
                self.input_shape, name="input_layer")
            x = bottom_layers(inputs)
            x = core_model(x)
            outputs = top_layers(x)
            return tf.keras.Model(inputs, outputs)
        else:
            return core_model
