import os
import logging
import glob

import hydra
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader

from trainer import LitModel
from data.dataset import MusicDataset

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def preprocess_dataset(dataset_folder: str, seq_len: int, batch_size: int, num_workers: int, *, overfit_single_batch: bool = False):
    filepaths = glob.glob(f"{dataset_folder}/**/**.mp3")

    train_ds = MusicDataset(filepaths, length=seq_len)

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_dataloader

@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    wandb_token = os.environ["WANDB_TOKEN"]

    wandb.login(key=wandb_token)

    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    logging.info("Preprocessing dataset...")
    # dataset
    train_dataloader = preprocess_dataset(
        dataset_folder=cfg.train.dataset_name,
        seq_len=cfg.train.seq_len,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # device = torch.device(cfg.train.device)

    logging.info("Initializing models...")
    lit_model = LitModel(cfg.var, nn.CrossEntropyLoss(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # checkpoint save path
    num_params_millions = sum([p.numel() for p in lit_model.parameters()]) / 1_000_000
    # save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}-params-{num_params_millions:.2f}M.ckpt"


    logging.info("Training...")

    logger = WandbLogger(
       project="music-autoregressive-modelling", 
        name=cfg.logger.run_name, 
        dir=cfg.paths.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True), 
    )

    callbacks = [ModelCheckpoint(
        dirpath=cfg.paths.save_ckpt_dir, 
        filename=f"{cfg.logger.run_name}-params-{num_params_millions:.2f}M.ckpt",
        every_n_train_steps=cfg.logger.log_every_n_steps,
    )]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        max_epochs=cfg.train.num_epochs,
    )

    trainer.fit(lit_model, train_dataloader)


if __name__ == "__main__":
    train()
