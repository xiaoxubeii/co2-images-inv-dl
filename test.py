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
            m["model_res_path"], test_dataset_path, m["config"])
        # model = model_eval.get_inversion_model_from_weights(
        #     m["model_res_path"], name_w=m["model_weights_name"], cfg=m["config"])
        model = model_eval.get_inversion_model_from_weights(
            m["model_res_path"], name_w=m["model_weights_name"])

        metric = model_eval.get_inv_metrics_model_on_data(
            model, data, sample_num=m["sample_num"])
        metric["method"] = m["method"]
        metrics.append(metric)
    model_eval.get_summary_histo_inversion1(metrics)
    plt.show()


# test_dataset = os.path.join(
#     config["data.path.directory"], config["data.path.train.name"], "test_dataset.nc")
# download_model(config["apikey"], run_path)
# run_path = "/kaggle/working/res/transformer/2024-06-13_21-10-26"
# test_exp("/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/best_essen_none",
#          "w_best.weights.h5", "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc")

window_length = 12
shift = 1
sample_num = 80
model1 = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/co2emissiontransformer/co2et-window12-patch16-64",
    "model_weights_name": "w_best.keras",
    "method": "co2et-window12-patch16-64",
    "sample_num": sample_num,
    "config": {
        "data": {"path": {"directory": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp"},
                 "init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-64/w_best.keras"}
    },
}

model2 = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/co2emissiontransformer/co2et-window12-patch16-chan5-64",
    "model_weights_name": "w_best.keras",
    "method": "co2et-window12-patch16-chan5-64",
    "sample_num": sample_num,
    "config": {
        "data": {"path": {"directory": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp"},
                 "init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-chan5-64/w_best.keras"}
    },
}
model3 = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/co2emissiontransformer/co2et-win84-patch16-64",
    "model_weights_name": "w_best.keras",
    "method": "co2et-win84-patch16-64",
    "sample_num": sample_num,
    "config": {
        "data": {"path": {"directory": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp"},
                 "init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-chan5-64/w_best.keras"}
    },
}
model_test = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/co2emission_transformer/co2emission_transformer_2024-06-27_12-49-13",
    "model_weights_name": "w_best.keras",
    "method": "co2et-win84-patch16-64",
    "sample_num": sample_num,
    "config": {
        "data": {"path": {"directory": "/Users/xiaoxubeii/Downloads/data_paper_inv_pp"},
                 "init": {"window_length": window_length, "shift": shift}, },
        "model": {"embedding_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/xco2transformer/xco2t-small-patch16-chan5-64/w_best.keras"}
    },
}

model_essen = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/essential/chan_none_epoch_1000",
    "model_weights_name": "w_best.weights.h5",
    "method": "essential",
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


# models = [model2, model3]
models = [model_test]
for m in models:
    with open(os.path.join(m["model_res_path"], "config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
        config = update(config, m["config"])
        m["config"] = OmegaConf.create(yaml.dump(config))

compare_exps(models,
             "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc")
