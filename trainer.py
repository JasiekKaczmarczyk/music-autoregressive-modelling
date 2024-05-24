import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math
import torchmetrics.functional as M
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

        self.model = VAR(**var_cfg).to(self.device)

        vae_path = dac.utils.download(model_type="44khz")
        self.vae = dac.DAC.load(vae_path).to(self.device)

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
        codes = []
        latents = []

        for scale in x.values():
            scale = scale.to(self.device)
            scale = self.preprocess(scale)
            _, code, _, _, _= self.vae.encode(scale)

            _, latent, _ = self.vae.quantizer.from_codes(code)
            # shape: [B L C]
            latent = latent.permute(0, 2, 1)

            codes.append(code)
            latents.append(latent)

        # codes shape: [B Q L]
        codes = torch.cat(codes, dim=-1)
        # latents shape: [B L Q*C]
        latents = torch.cat(latents, dim=1)

        return latents, codes

    
    def training_step(self, batch: tuple[dict[torch.Tensor], torch.Tensor], batch_idx: int):
        """

        Args:
            batch (tuple[dict[torch.Tensor], torch.Tensor]): contains signal dict (Hz: signal) and condition embedding
            batch_idx (int): idx

        Returns:
            torch.Tensor: loss
        """
        signals, cond = batch
        cond = cond.to(self.device)
        latents, codes = self.prepare_batch(signals)
        logits = self.model(latents, cond)

        # loss = self.criterion(logits_wo_start_token.view(-1, C), codes.view(-1))
        loss = self.criterion(logits.view(-1, self.vae.codebook_size), codes.view(-1))
        accuracy = (codes == torch.argmax(logits, dim=-1))

        metrics = {
            "train/loss": loss.item(),
            "train/accuracy": accuracy.item(),
        }

        self.log_dict(metrics)

        return loss

if __name__ == "__main__":
    from data.dataset import MusicDataset
    import glob
    from torch.utils.data import DataLoader
    ds = glob.glob("music/**.mp3")

    dataset = MusicDataset(ds, length=1_048_576)
    # sample = dataset[0]
    loader = DataLoader(dataset, batch_size=8)

    signals, y = next(iter(loader))

    cfg = VARConfig(latent_size=72, hidden_size=128, cond_size=128, out_size=1024, num_spatial_layers=4, num_depth_layers=4, num_heads=4)

    model = LitModel(cfg, criterion=nn.CrossEntropyLoss()).to("cuda")

    # print(codes.shape)
    # print(latents.shape)

    loss = model.training_step((signals, y), 0)
    print(loss)

    # print(model.training_step((x, y), 0))