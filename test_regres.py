
import hydra
from omegaconf import DictConfig, OmegaConf
import model_eval
import os
import wandb
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Data import Data_eval
import matplotlib_functions as mympf
import keras


if __name__ == "__main__":
    cfg = OmegaConf.create(
        [{"name": "boxberg", "nc": "test_dataset.nc"}]
    )
    data = Data_eval(
        "/Users/xiaoxubeii/Downloads/data_paper_inv_pp", cfg, 0, 0)
    data.prepare_input(
        "xco2",
        "u_wind",
        "v_wind"
    )

    model = keras.models.load_model(
        "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/co2emiss-regres-08030008/w_best.keras")
    model.summary()
