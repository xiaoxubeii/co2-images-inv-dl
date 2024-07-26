# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from model_training import Model_training_manager
import tensorflow as tf


@hydra.main(config_path="cfg", config_name="config")
def main_train(cfg: DictConfig):
    print("\n \n \n \n \n Run begins \n \n \n \n \n")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # with strategy.scope():
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    # instantiate a distribution strategy
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)
    with tpu_strategy.scope():
        model_trainer = Model_training_manager(cfg)
    val_loss = model_trainer.run()
    model_trainer.save()
    print("\n \n \n \n \n Run ends \n \n \n \n \n")
    return val_loss


if __name__ == "__main__":
    main_train()
