# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
from tensorflow import keras


def define_optimiser(optimiser_name: str = "adam", learning_rate: float = 1e-3):
    """Define optimiser with learning rate."""

    dicOpt = {
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
    }

    opt = dicOpt[optimiser_name]

    return opt

def define_lr_scheduler():
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
