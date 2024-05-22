import os
import logging
import glob

import hydra
import torch
import wandb
import torch.nn as nn
import dac
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import torchaudio

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader

from models.autoencoder import MultiScaleDAC
from models.var import VAR, VARConfig
from data.dataset import MusicDataset

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_folder: str, seq_len: int, batch_size: int, num_workers: int, *, overfit_single_batch: bool = False):
    filepaths = glob.glob(f"{dataset_folder}/**.mp3")

    train_ds = MusicDataset(filepaths, length_seconds=seq_len)

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_dataloader

def save_checkpoint(
    model: VAR, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str
):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


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

    # logger
    wandb.init(
        project="music-autoregressive-modelling", 
        name=cfg.logger.run_name, 
        dir=cfg.paths.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    device = torch.device(cfg.train.device)

    logging.info("Initializing models...")
    # Download a model
    dac_path = dac.utils.download(model_type="44khz")
    teacher = dac.DAC.load(dac_path).to(device)

    # model
    model = MultiScaleDAC(n_scales=24, codebook_dim=16).to(device)
    model.encoder.load_state_dict(teacher.encoder.state_dict())
    model.decoder.load_state_dict(teacher.decoder.state_dict())

    logging.info("Optimizers...")

    # setting up optimizer
    optimizer = optim.AdamW(model.quantizer.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # checkpoint save path
    num_params_millions = sum([p.numel() for p in model.parameters()]) / 1_000_000
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}-params-{num_params_millions:.2f}M.ckpt"

    # step counts for logging to wandb
    step_count = 0

    signal_distortion_ratio = ScaleInvariantSignalDistortionRatio().to(device)

    logging.info("Training...")

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        model.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        loss_epoch = 0.0

        for batch_idx, batch in train_loop:
            audio = batch["audio_data"].to(device)

            with torch.no_grad():
                z = teacher.encoder(audio)
                target, _, _, _, _ = teacher.quantizer(z)

            pred, _ = model.quantizer(z)

            loss = F.mse_loss(pred, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reconstructed = model.decode(pred)
                length = reconstructed.shape[-1]
                reconstruction_loss = F.mse_loss(reconstructed, audio[:, :, :length])
                psnr = 10 * torch.log10((torch.amax(audio) ** 2) / reconstruction_loss)
                sdr = signal_distortion_ratio(reconstructed, audio[:, :, :length])

            metrics = dict(
                latent_loss=loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                psnr=psnr.item(),
                sdr=sdr.item(),
            )
            train_loop.set_postfix(metrics)

            step_count += 1
            loss_epoch += loss.item()

            if (step_count + 1) % cfg.logger.log_every_n_steps == 0:

                # log metrics
                wandb.log(metrics, step=step_count)

                torchaudio.save(f"generated/step_{step_count}.mp3", reconstructed[0].cpu(), sample_rate=44_100)

                # save model and optimizer states
                save_checkpoint(model, optimizer, cfg, save_path=save_path)

        training_metrics = {"train/loss_epoch": loss_epoch / len(train_dataloader)}

        wandb.log(training_metrics, step=step_count)

    # save model at the end of training
    save_checkpoint(model, optimizer, cfg, save_path=save_path)

    wandb.finish()


if __name__ == "__main__":
    train()
