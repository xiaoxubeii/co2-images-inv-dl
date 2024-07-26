import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


@hydra.main(config_path="cfg", config_name="config")
def main_train(cfg: DictConfig):

    # Detect TPU, return appropriate distribution strategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)


if __name__ == "__main__":
    main_train()
