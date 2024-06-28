import collections.abc
import pdb
import model_eval
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf


def compare_exps(models, test_dataset_path):
    metrics = []
    for m in models:
        data = model_eval.get_data_for_inversion(
            os.path.join(m["model_res_path"], m["model_type"], m["model_name"]), test_dataset_path, m["config"])
        model = model_eval.get_inversion_model_from_weights(
            os.path.join(m["model_res_path"], m["model_type"], m["model_name"]), name_w=m["model_weights_name"], cfg=m["config"])

        metric = model_eval.get_inv_metrics_model_on_data(
            model, data, sample_num=m["sample_num"])
        metric["method"] = m["model_name"]
        metrics.append(metric)
    model_eval.get_summary_histo_inversion1(metrics)
    plt.show()


# test_dataset = os.path.join(
#     config["data.path.directory"], config["data.path.train.name"], "test_dataset.nc")
# download_model(config["apikey"], run_path)
# run_path = "/kaggle/working/res/transformer/2024-06-13_21-10-26"
# test_exp("/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/best_essen_none",
#          "w_best.weights.h5", "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc")

window_length = 6
shift = 1
sample_num = 160
data_dir = "/Users/xiaoxubeii/Downloads/data_paper_inv_pp"
res_dir = ""
model1 = {
    "model_type": "co2emissiontransformer",
    "model_weights_name": "w_best.keras",
    "model_name": "co2et-window12-patch16-64",
    "sample_num": sample_num,
    "config": {
        "data": {"init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-64/w_best.keras"}
    },
}

model2 = {
    "model_type": "co2emissiontransformer",
    "model_weights_name": "w_best.keras",
    "model_name": "co2et-window12-patch16-chan5-64",
    "sample_num": sample_num,
    "config": {
        "data": {"init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-chan5-64/w_best.keras"}
    },
}
model3 = {
    "model_type": "co2emissiontransformer",
    "model_weights_name": "w_best.keras",
    "model_name": "co2et-win84-patch16-64",
    "sample_num": sample_num,
    "config": {
        "data": {"init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-64/w_best.keras"}
    },
}
model4 = {
    "model_type": "co2emissiontransformer",
    "model_weights_name": "w_best.keras",
    "model_name": "co2et-win12-patch8-64",
    "sample_num": sample_num,
    "config": {
        "data": {"init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch8-64/w_best.keras"}
    },
}

model_cnn = {
    "model_type": "cnn",
    "model_weights_name": "w_best.weights.h5",
    "model_name": "cnn-baseline",
    "sample_num": sample_num*window_length,
    "config": {
        "data": {"path": {"directory": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp"}}
    }
}


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


models = [model1, model2, model3, model4, model_cnn]
# models = [model4]
# models = [model3]
for m in models:
    m = update(m, {
        "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments", "config": {"data": {"path": {"directory": data_dir}}}})

    with open(os.path.join(m["model_res_path"], m["model_type"], m["model_name"], "config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
        config = update(config, m["config"])
        m["config"] = OmegaConf.create(yaml.dump(config))

compare_exps(models,
             "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc")
