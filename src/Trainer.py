import gc
import os
import random
from typing import Any, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from BaseTrainer import BaseDistributedTrainer, State

class Trainer(BaseDistributedTrainer):
    def __init__(self, config: DictConfig) -> None:
        # set local rank and random seed
        super(Trainer, self).__init__(config)
    
    def init_dataset(config: DictConfig):
        raise NotImplementedError

    def init_dataloader(config: DictConfig):
        raise NotImplementedError

    def init_model(config: DictConfig):
        raise NotImplementedError

    def init_summaryWriter(config: DictConfig):
        raise NotImplementedError

    def train():
        raise NotImplementedError