
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

    path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/co2emiss-regres-08030211"
    model = keras.models.load_model(f"{path}/w_best.keras")

    model = model.get_layer("co2emiss_regres")
    embedding = keras.Model(model.input, model.layers[-4].output)
    quantifying = keras.Model(model.layers[-4].output, model.output)

    result = embedding(data.x.eval_data)
    print(result)
