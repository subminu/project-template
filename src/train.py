import gc
import os
from typing import Tuple, Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src import utils

LOG = utils.get_logger(__name__)


@utils.set_dist_func
def train(config: DictConfig, local_rank: int) -> None:
    # Set seed for reproduction
    if config.get("rand_seed"):
        utils.set_seed(config.rand_seed)

    # Initiate model
    if config.get("checkpoint_file") and os.path.isfile(config.checkpoint_file):
        LOG.info("Load a trained model.")
        model, criterion, optimizer = load_model(
            config.checkpoint_file, config.model, local_rank
        )
    else:
        LOG.info("There is no trained model, Initiate a new model.")
        model, criterion, optimizer = init_model(config.model, local_rank)

    # Initiate dataloaders
    train_loader, valid_loader = init_data_loader(config.dataset, config.data_loader)
    # Initiate a tensorboard (only zero rank)
    writer = hydra.utils.instantiate(config.logger) if local_rank == 0 else None

    dist.barrier()
    LOG.info("All ranks are ready to train.")

    for epoch in range(config.max_epochs):
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        for batch in tqdm(train_loader):
            image, label = get_batch(batch, local_rank)
            # Compute output
            with autocast():
                outputs = model(image)
                loss = criterion(outputs, label)

            # Compute gradient & optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for data in tqdm(valid_loader):
                image, label = get_batch(data, local_rank)

                # Compute output
                outputs = model(image)
                loss = criterion(outputs, label)

        gc.collect()
        torch.cuda.empty_cache()

    if writer:
        writer.close()
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "model.pt"))
        LOG.info("Saved model.")

    LOG.info("Finished train.")


def get_batch(
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], local_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    image, label = data
    return image.to(local_rank), label.to(local_rank)


def load_model(
    checkpoint_file: str, config: DictConfig, local_rank: int
) -> Tuple[Any, Any, Any]:
    # model
    model, criterion, optimizer = init_model(config, local_rank)

    model.load_state_dict(
        torch.load(checkpoint_file, map_location=f"cuda:{local_rank}")
    )

    return model, criterion, optimizer


def init_model(config: DictConfig, local_rank: int) -> Tuple[Any, Any, Any]:
    # model
    model = hydra.utils.instantiate(config.architecture).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    return model, criterion, optimizer


def init_data_loader(
    config_dataset: DictConfig, config_loader: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    dataset = hydra.utils.instantiate(config_dataset.dataset)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - 1000, 1000],
        generator=torch.Generator().manual_seed(42),
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    valid_sampler = DistributedSampler(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config_loader.train.batch_size,
        pin_memory=config_loader.train.pin_memory,
        persistent_workers=config_loader.train.persistent_workers,
        num_workers=config_loader.train.num_workers,
        prefetch_factor=config_loader.train.prefetch_factor,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=config_loader.val.batch_size,
        persistent_workers=config_loader.train.persistent_workers,
        num_workers=config_loader.train.num_workers,
    )

    return train_dataloader, valid_dataloader
