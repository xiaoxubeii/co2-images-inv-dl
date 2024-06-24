import model_eval
import numpy as np
import os
import matplotlib.pyplot as plt


def compare_exps(models, test_dataset_path):
    metrics = []
    for m in models:
        data = model_eval.get_data_for_inversion(
            m["model_res_path"], test_dataset_path)
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
model1 = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/transformer/emiss_tran_5",
    "model_weights_name": "w_best.weights.h5",
    "method": "emiss_trans",
    "sample_num": 10
}
model2 = {
    "model_res_path": "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/best_essen_none",
    "model_weights_name": "w_best.weights.h5",
    "method": "essential",
    "sample_num": 10*84
}
compare_exps([model1, model2],
             "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc")
