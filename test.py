import model_eval
import numpy as np
import os
import matplotlib.pyplot as plt


def test_exp(model_res_path, model_weights_name, test_dataset_path):
    data = model_eval.get_data_for_inversion(model_res_path, test_dataset_path)
    model = model_eval.get_inversion_model_from_weights(
        model_res_path, name_w=model_weights_name)
    metrics = model_eval.get_inv_metrics_model_on_data(model, data)
    print("mae:", np.mean(metrics["mae"]))
    print("mape:", np.mean(metrics["mape"]))

    model_eval.plot_inversion_examples(
        data, metrics["mae"], metrics["mape"], model)
    model_eval.get_summary_histo_inversion(model, data)
    plt.show()


# test_dataset = os.path.join(
#     config["data.path.directory"], config["data.path.train.name"], "test_dataset.nc")
# download_model(config["apikey"], run_path)
# run_path = "/kaggle/working/res/transformer/2024-06-13_21-10-26"
test_exp("/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/inversion/best_essen_none",
         "w_best.weights.h5", "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc")
