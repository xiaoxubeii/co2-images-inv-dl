import models.reg as rm
import model_eval
import matplotlib_functions as mympf
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr
from cmcrameri import cm
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from scipy.optimize import differential_evolution
from sklearn import preprocessing
import os

sys.path.append(
    "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl")
# from Data import Data_eval

mympf.setMatplotlibParam()
plt.viridis()
dir_res = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/inv_lip_cnn_lstm"
name_model = dir_res
path_eval_nc = "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/all_hotspots_but_lip/train_dataset.nc"
data = model_eval.get_data_for_inversion(
    dir_res,
    path_eval_nc,
)

model = model_eval.get_inversion_model(
    os.path.join(dir_res, name_model), name_w="w_last.keras"
)
metrics = model_eval.get_inv_metrics_model_on_data(model, data)
print("mae:", np.mean(metrics["mae"]))
print("mape:", np.mean(metrics["mape"]))

model_eval.plot_inversion_examples(data, metrics["mae"], metrics["mape"], model)
model_eval.plot_inversion_examples(data, metrics["mae"], metrics["mape"], model)

plt.show()  