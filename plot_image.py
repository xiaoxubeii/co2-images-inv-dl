import pdb
from Data import Data_eval
from omegaconf import OmegaConf
import keras
from models.reg import BottomLayers
from plot_data import plot_examples
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_functions as mympf

cfg = OmegaConf.create(
    [{"name": "boxberg", "nc": "test_dataset.nc"}]
)
data = Data_eval("/Users/xiaoxubeii/Downloads/data_paper_inv_pp", cfg, 0, 0)
data.prepare_input(
    "xco2",
    "u_wind",
    "v_wind"
)

# embedding = keras.models.load_model(
#     "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/embedding/xco2embedd-mae-patch16/w_best.keras")
embedding = keras.models.load_model(
    "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/embedding/xco2embedd-mae-patch8/w_best.keras")

embedding.patch_encoder.downstream = True

pretrained_embedding_layer = keras.Sequential([
    embedding.patch_layer,
    embedding.patch_encoder,
    embedding.encoder,
    embedding.decoder,
], name="embedding")

bottom_layers = BottomLayers(
    data.x.n_layer, data.x.eval_data.shape[-1], data.x.xco2_noisy_chans, 0)

inputs = bottom_layers(data.x.eval_data)
inputs = inputs[0:1]
outputs = pretrained_embedding_layer(inputs)


onorm = keras.layers.Normalization(axis=-1, invert=True)
onorm.adapt(data.x.eval_data)

# print(keras.losses.MeanSquaredError()(inputs, outputs))
outputs = onorm(outputs)


def plot_examples(
    input_shape,
    data,
    list_idx,
):
    """Plot examples of {input / truth / output} of the CNN model."""
    N_idx = len(list_idx)
    N_cols = 3
    mympf.setMatplotlibParam()
    plt.viridis()
    fig, axs = mympf.set_figure_axs(
        N_idx,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.001,
        pad_h_ext=0.2,
        pad_h_int=0.25,
    )

    ims = [None] * (N_idx * N_cols)
    caxs = [None] * (N_idx * N_cols)
    cbars = [None] * (N_idx * N_cols)

    Ny = int(input_shape[0])
    Nx = int(input_shape[1])
    for ax in axs:
        ax.set_xticks([0, int(Ny / 4), int(Ny / 2), int(Ny * 3 / 4), Ny])
        ax.set_yticks([0, int(Nx / 4), int(Nx / 2), int(Nx * 3 / 4), Nx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def _plot_data(data, list_idx):
        for i, idx in enumerate(list_idx):
            cur_row = 0
            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data[idx, :, :, 0]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data[idx, :, :, 1]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data[idx, :, :, 2]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

    _plot_data(data, list_idx)


data = np.array([inputs[0], outputs[0]])
plot_examples(inputs[0].shape, data, list_idx=[0, 1])
plt.show()