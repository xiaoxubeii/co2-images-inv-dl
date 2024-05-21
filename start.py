import sys
import os
import uuid
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


def test_exp(exp_name, res_path, test_dataset):
    cfg = OmegaConf.load(os.path.join(res_path, exp_name, "config.yaml"))
    data = model_eval.get_data_for_inversion(
        os.path.join(res_path, exp_name), test_dataset,
    )
    model = model_eval.get_inversion_model(
        os.path.join(res_path, exp_name), name_w="w_last.keras")
    # metrics = model_eval.get_inv_metrics_model_on_data(model, data)
    # print("mae:", np.mean(metrics["mae"]))
    # print("mape:", np.mean(metrics["mape"]))
    # model_eval.plot_inversion_examples(
    #     data, metrics["mae"], metrics["mape"], model, window_length=cfg["data"]["init"]["window_length"])

    model_eval.get_summary_histo_inversion(model, data)


source_code = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl"
default_params = {
    "source_code": source_code,
    "experiment": "inv",
    "res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion",
    "max_epochs": 10,
    "window_length": 0,
    "shift": 0,
    "batch_size": 32,
    "model_name": "",
    "dataset_path": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/",
    "train_dataset": "all_hotspots_but_box_reduced",
    "valid_dataset": "all_hotspots_but_box_reduced",
    "test_dataset": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/all_hotspots_but_box_reduced/test_dataset.nc",
    "exp_name": "",

}


def default(**args):
    r = default_params.copy()
    for k, v in args.items():
        r[k] = v
    return r


def run_exp(**args):
    run_exp = "python {source_code}/main.py +experiment={experiment} ++exp_name={exp_name} ++data.path.directory={dataset_path} ++data.path.train.name={train_dataset} ++data.path.valid.name={valid_dataset} ++training.max_epochs={max_epochs} ++data.init.window_length={window_length} ++data.init.shift={shift} ++training.batch_size={batch_size} ++model.name={model_name}"
    config = default(**args)
    run_exp = run_exp.format(**config)
    os.system(run_exp)


# run_exp(exp_name="cnn-lstm",
#         model_name="cnn-lstm", window_length=2, shift=1, max_epochs=1)

test_exp(exp_name="cnn-lstm",
         res_path=default_params["res_path"], test_dataset=default_params["test_dataset"])
plt.show()
