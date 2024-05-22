import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math
import einops
from omegaconf import DictConfig

from models.var import VAR, VARConfig
import dac

class CosSimLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()

        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        B, L, C = pred.shape

        pred = F.normalize(pred, dim=-1).view(-1, C)
        target = F.normalize(target, dim=-1).view(-1, C)

        # scaled pairwise cosine similarities
        # logits = einops.einsum(pred, target, "n l c, m k c -> (n l) (m k)")
        logits = torch.einsum("n c, m c -> n m", [pred, target])
        logits = logits * math.exp(self.temperature)

        labels = torch.arange(0, B * L, dtype=torch.long, device=pred.device)

        loss = F.cross_entropy(logits, labels)
        # loss_per_pitch = F.cross_entropy(logits.t(), labels)

        # get loss value for batch
        # loss = (loss_per_velocity_time + loss_per_pitch) / 2

        return loss

class LitModel(pl.LightningModule):
    def __init__(self, var_cfg: DictConfig, criterion: nn.Module, lr: float = 3e-4, weight_decay: float = 0.001):
        super().__init__()

        self.model = VAR(**var_cfg)

        vae_path = dac.utils.download(model_type="44khz")
        self.vae = dac.DAC.load(vae_path)

        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
    
    def preprocess(self, audio_data: torch.Tensor):

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.vae.hop_length) * self.vae.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data
    
    @torch.no_grad()
    def prepare_batch(self, x: dict[torch.Tensor]):
        # codes: [B Q L]
        codes = []
        # latents: [B L Q C]
        latents = []

        for scale in x.values():
            scale = self.preprocess(scale)
            _, code, _, _, _= self.vae.encode(scale)

            codes.append(code)

        return codes, latents

    
    def training_step(self, batch: tuple[dict[torch.Tensor], torch.Tensor], batch_idx: int):
        """

        Args:
            batch (tuple[dict[torch.Tensor], torch.Tensor]): contains signal dict (Hz: signal) and condition embedding
            batch_idx (int): idx

        Returns:
            torch.Tensor: loss
        """
        x, y = batch

        x, _ = self.prepare_batch(x)

        # shape: [B L C]
        x = x.permute(0, 2, 1)
        
        logits = self.model(x, y)
        logits_wo_start_token = logits[:, 1:, :]
        # B, L, C = logits_wo_start_token.shape

        # loss = self.criterion(logits_wo_start_token.view(-1, C), codes.view(-1))
        loss = self.criterion(logits_wo_start_token, x)

        metrics = {"train/loss": loss}

        self.log_dict(metrics)

        return loss

if __name__ == "__main__":
    x = torch.randn((1, 1, 200_000))
    y = torch.randn((1, 128))

    cfg = VARConfig(latent_size=1024, hidden_size=128, cond_size=128, depth=4, num_heads=4)

    model = LitModel(cfg, criterion=CosSimLoss())

    # z, idx = model.prepare_batch(x)

    # print(z.shape)
    # print(idx.shape)

    print(model.training_step((x, y), 0))