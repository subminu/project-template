import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    from src.train import train

    return train(config)


if __name__ == "__main__":
    main()
