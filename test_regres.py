
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
from plot_image import plot_image_from_mae


if __name__ == "__main__":
    cfg = OmegaConf.create(
        [{"name": "boxberg", "nc": "test_dataset.nc"}]
    )
    data = Data_eval(
        "/Users/xiaoxubeii/Downloads/data_paper_inv_pp", cfg, 2, 1)
    data.prepare_input(
        "xco2",
        "u_wind",
        "v_wind"
    )

    # path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/embedding/xco2embedd-mae-patch16"
    # model = keras.models.load_model(f"{path}/w_best_1.keras")
    # model = model.get_layer("co2emiss_regres")
    # # embedding = keras.Model(model.input, model.layers[-5].output)
    # model.get_layer("patch_encoder").downstream = True
    # embedding = keras.Sequential([
    #     model.get_layer("patches"),
    #     model.get_layer("patch_encoder"),
    #     model.get_layer("mae_encoder"),
    # ])
    # path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/inversion/co2emiss-regres-patch16/w_best.keras"
    # regres_model = keras.models.load_model(path)

    # path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/co2emiss-transformer-08041353/w_best.keras"
    # transf_model = keras.models.load_model(path)

    # model = keras.Sequential([
    #     transf_model,
    #     regres_model
    # ])

    # print(model.predict(data.x.eval_data[data.x.eval_data_indexes]))

    # plot_image_from_mae(data, mae)

    # path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/co2emiss-transformer-08050033/w_best.keras"
    # model = keras.models.load_model(path)
    # print(model.predict(data.x.eval_data[data.x.eval_data_indexes]))
    path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/co2emiss-transformer-08050135/w_best.keras"
    model = keras.models.load_model(path)
    print(model.predict(data.x.eval_data[data.x.eval_data_indexes]))
