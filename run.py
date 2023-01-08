import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./conf", config_name="main")
def main(config: DictConfig):
    from src.train import train

    return train(config)


if __name__ == "__main__":
    main()
