import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dac.nn.layers import WNConv1d
from dac.model.dac import ResidualUnit
# from models.lfq import LFQ
from models.fsq import FSQ

class VARQuantizer(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_dim: int = 16,
        n_scales: int = 24,
        scale_levels: list[int] = [1, 2, 4, 8, 10, 12]
    ):
        super().__init__()

        self.codebook_dim = codebook_dim
        assert n_scales % len(scale_levels) == 0
        mult = n_scales // len(scale_levels)

        self.scales = list(reversed(2 ** np.repeat(scale_levels, mult)))

        self.in_proj = nn.Sequential(
            ResidualUnit(input_dim),
            WNConv1d(input_dim, codebook_dim, kernel_size=1),
        )

        # self.quantizers = LFQ(
        #     dim=codebook_dim,
        # )
        self.quantizers = FSQ(
            levels=[8, 8, 8, 8, 5],
            dim=codebook_dim
        )

        self.phi = ResidualUnit(codebook_dim)
        self.out_proj = nn.Sequential(
            WNConv1d(codebook_dim, input_dim, kernel_size=1),
            ResidualUnit(input_dim),
        )

    def forward(self, z: torch.Tensor):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "aux_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
        """
        z = self.in_proj(z)
        z_q = torch.zeros_like(z)

        codes = []

        z_length = z.shape[-1]

        for scale in self.scales:

            scaled_length = z_length // scale

            
            z_downsampled = F.interpolate(z, size=scaled_length).permute(0, 2, 1)
            # returns (z_q, indices, codebook_loss)
            r, indices = self.quantizers(z_downsampled)
            
            codes.append(indices)
            z_q_upsampled = F.interpolate(r.permute(0, 2, 1), size=z_length)
            z_q_phi = self.phi(z_q_upsampled)

            z -= z_q_phi
            z_q += z_q_phi

        z_q = self.out_proj(z_q)

        return z_q, codes


if __name__ == "__main__":
    q = VARQuantizer(input_dim=4, codebook_dim=4, n_scales=24)
    x = torch.randn(1, 4, 4096)
    x_q, codes, loss = q(x)
    print(x)
    print(x_q)
    
