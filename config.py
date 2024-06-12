import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def load_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    load_config()
