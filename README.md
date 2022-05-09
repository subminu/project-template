# Project Template
This project template is for deep-learning researchers who want to use multi-gpu with pytorch Distributed Data Parallel(DDP).

## Pre-requsites
This repository ensure that works in python 3.8 or later installed all `requirements.txt` dependencies. Additionally, this project-template uses configuration manager framework, [Hydra](https://hydra.cc/). If you are not familier with Hydra, please check this [Hydra tutorial docs](https://hydra.cc/docs/intro/).

```bash
pip install -r requirements.txt
```

## Folder tree
```text
Project-Name/
├── configs/ # Hydra configuration files goes here
│   ├── data_loader/ # data_loader configs
│   ├── dataset/ # dataset configs
│   ├── log_dir/ # directory configs to save all logs during training
│   ├── logger/ # visualization tool configs
│   ├── model/ # model configs
│   └── default.yaml # main config
│
├── data/ # all dataset goes here
│
├── logs/ # all logs goes here
│
├── src/ # source codes goes here
│   ├── dataloaders/ 
│   ├── datasets/
│   ├── models/
│   ├── utils/ # util functions for multi-gpu
│   └── train.py 
│
└── run.py # you can train the model by run this code
```

## How to run
Please set the develop environment python 3.8 or later version and install all dependencies.

```bash
pip install -r requirements.txt
```

This project use multi-gpu by using elastic launch ([torchrun](https://pytorch.org/docs/stable/elastic/run.html)), if not familiar with Torchrun, please check this [documenation](https://pytorch.org/docs/stable/distributed.elastic.html).

```bash
torchrun --nproc_per_node <num_gpu> run.py
```

## Acknowledgement
This project-template is inspired by the project [Pytorch-Lightning-Template](https://github.com/ashleve/lightning-hydra-template) and [Pytorch-elastic-examples](https://github.com/pytorch/elastic/tree/master/examples).

## License
This project is licensed under [MIT License](LICENSE).