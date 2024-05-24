import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp
from torch.jit import Final
from omegaconf import DictConfig


def modulate(x, shift, scale):
    return x * (1 * scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, 
                k, 
                v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn += attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttnBlock(nn.Module):
    """
    A attention block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_bias: torch.Tensor = None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        if x.shape[0] != shift_msa.shape[0]:
            num_repeats = x.shape[0] // shift_msa.shape[0]
            shift_msa = torch.repeat_interleave(shift_msa, repeats=num_repeats, dim=0)
            scale_msa = torch.repeat_interleave(scale_msa, repeats=num_repeats, dim=0)
            gate_msa = torch.repeat_interleave(gate_msa, repeats=num_repeats, dim=0)

            shift_mlp = torch.repeat_interleave(shift_mlp, repeats=num_repeats, dim=0)
            scale_mlp = torch.repeat_interleave(scale_mlp, repeats=num_repeats, dim=0)
            gate_mlp = torch.repeat_interleave(gate_mlp, repeats=num_repeats, dim=0)


        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_bias)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer.
    """
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)

        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class VAR(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        latent_size: int = 128,
        cond_size: int = 128,
        hidden_size: int = 1152,
        out_size: int = 1024,
        scales: list[int] = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        num_spatial_layers: int = 28,
        num_depth_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_codebooks: int = 9,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.scales = scales
        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks

        self.first_l = scales[0]
        self.sequence_length = sum(scales)

        self.x_embedder = nn.Linear(latent_size, hidden_size)
        self.cond_embedder = nn.Linear(cond_size, hidden_size)

        # position embedding
        # self.pos_embed = nn.Parameter(torch.zeros(1, max_sequence_length, hidden_size), requires_grad=False)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, hidden_size))
        self.pos_emb = nn.Parameter(torch.empty(1, self.sequence_length, hidden_size))
        self.level_emb = nn.Embedding(len(scales), hidden_size)

        self.spatial_transformer = nn.ModuleList([
            AttnBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio
            )
            for _ in range(num_spatial_layers)
        ])

        # attention mask
        d = torch.cat([torch.full((scale,), i) for i, scale in enumerate(scales)]).view(1, self.sequence_length, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.sequence_length, self.sequence_length)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # output
        self.depth_emb = nn.Parameter(torch.empty(1, num_codebooks + 1, hidden_size))
        self.codebook_proj = nn.Linear(latent_size // num_codebooks, hidden_size)

        self.depth_transformer = nn.ModuleList([
            AttnBlock(
                hidden_size,
                num_heads, 
                mlp_ratio=mlp_ratio
            )
            for _ in range(num_depth_layers)
        ])

        attn_bias_depth = torch.tril(torch.ones((num_codebooks+1, num_codebooks+1)))
        attn_bias_depth.masked_fill(attn_bias_depth == 0, -torch.inf).reshape(1, 1, num_codebooks+1, num_codebooks+1)
        self.register_buffer('attn_bias_depth', attn_bias_depth.contiguous())

        self.final_layer = FinalLayer(hidden_size, out_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], torch.arange(self.max_sequence_length))
        # self.pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        init_std = math.sqrt(1 / self.hidden_size / 3)

        nn.init.trunc_normal_(self.pos_start, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pos_emb, mean=0, std=init_std)
        nn.init.trunc_normal_(self.level_emb.weight.data, mean=0, std=init_std)

        # Initialize x_embedder
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.spatial_transformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.depth_transformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        """
        Forward pass of DiT.
        x: (B, L, C) tensor of spatial inputs (images or latent representations of images)\
        depth_ctx: (B, L, codebook_size, C)
        cond: (B, C) tensor of class labels
        """
        B, L, _ = x.shape

        # shape: [B L Q latent_size // Q]
        depth_ctx = x.clone().reshape(B, L, self.num_codebooks, -1)

        x = self.x_embedder(x)
        cond = self.cond_embedder(cond)

        sos = self.pos_start.expand(B, self.first_l, -1) + cond.reshape(B, 1, -1)
        x = torch.cat([sos, x], dim=1)

        # adding level embedding and position embedding
        x += self.level_emb(self.lvl_1L)
        x += self.pos_emb

        # spatial transformer
        for block in self.spatial_transformer:
            x = block(x, cond, self.attn_bias_for_masking)

        # depth transformer
        depth_ctx = self.codebook_proj(depth_ctx)

        # drop start token and reshape
        spatial_ctx = x[:, 1:, :].reshape(B, L, 1, -1)
        # concatenating on codebook dimension
        depth_ctx = torch.cat([spatial_ctx, depth_ctx], dim=-2)
        depth_ctx = depth_ctx.reshape(B*L, -1, self.hidden_size) + self.depth_emb

        for block in self.depth_transformer:
            depth_ctx = block(depth_ctx, cond, self.attn_bias_depth)

        # drop spatial context and reshape
        depth_ctx = depth_ctx[:, 1:, :].reshape(B, L, -1, self.hidden_size)

        out = self.final_layer(depth_ctx, cond)

        return out

class VARConfig(DictConfig):
    def __init__(
            self,
            latent_size: int = 128,
            cond_size: int = 128,
            hidden_size: int = 1152,
            out_size: int = 1024,
            scales: list[int] = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            num_spatial_layers: int = 28,
            num_depth_layers: int = 4,
            num_heads: int = 16,
            mlp_ratio: float = 4.0,
            num_codebooks: int = 9,
        ):
        content = dict(
            latent_size=latent_size,
            cond_size=cond_size,
            hidden_size=hidden_size,
            out_size=out_size,
            scales=scales,
            num_spatial_layers=num_spatial_layers,
            num_depth_layers=num_depth_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_codebooks=num_codebooks,
        )

        super().__init__(content=content)
 
if __name__ == "__main__":
    x = torch.randn(2, 4092, 72).to("cuda")
    cond = torch.randn(2, 128).to("cuda")

    model = VAR(
        latent_size=72,
        cond_size=128,
        hidden_size=256,
        out_size=1024,
        num_spatial_layers=12,
        num_depth_layers=8,
        num_heads=4,
        mlp_ratio=4.0,
        num_codebooks=9,
    ).to("cuda")

    print(sum([p.numel() for p in model.parameters()]) / 1_000_000)

    print(model(x, cond).shape)

