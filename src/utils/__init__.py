import logging
import os
import random
import traceback
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig


# --------- decorator --------- #
def rank_zero_only(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def nested_func(*args, **kwargs) -> None:
        if int(os.environ["LOCAL_RANK"]) == 0:
            func(*args, **kwargs)

    return nested_func


def set_dist_func(func: Callable[[DictConfig, int], None]):
    def nested_func(*args, **kwarg) -> None:
        # Initicate the multi process group
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        try:  # If encounter some error, safely destroy the process group
            func(*args, local_rank, **kwarg)
        finally:
            torch.distributed.destroy_process_group()

    return nested_func

class SetDistibutedFunc(object):
    def __init__(self, error_logger):
        self.error_logger = error_logger
    
    def __call__(self, func: Callable[[DictConfig, int], None]):
        def _nested_func(*args, **kwarg) -> None:
            # Initicate the multi process group
            dist.init_process_group(backend="nccl", init_method="env://")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            try:  # If encounter some error, safely destroy the process group
                func(*args, local_rank, **kwarg)
            except Exception:
                self.error_logger.error(f"Some errors occurred. Please check the error the following log.\n{traceback.format_exc()}")
            finally:
                dist.destroy_process_group()
        return _nested_func

# --------- decorator --------- #


def set_seed(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger
