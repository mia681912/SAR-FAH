
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.0,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dropout=0.2,
            conv_bias=True,
    ):
        """
        input:B,L,D
        d_model: D, input channel
        d_state: A,B,C,D 的通道数，决定了实行mamba的维度
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # State space parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output layers
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.shape
        L = H * W
        K = 4  # Number of scanning directions
        dim = self.d_inner * K  # dim used for scanning

        # Project input
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, d_inner)

        # Conv activation
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x = self.act(self.conv2d(x))  # (B, d_inner, H, W)

        # Prepare scanning directions
        x_hwwh = torch.stack(
            [x.flatten(2), x.transpose(2, 3).flatten(2)],
            dim=1
        )  # (B, 2, d_inner, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, d_inner, L)

        # Project to get dt, B, C
        xs = xs.permute(0, 1, 3, 2)  # (B, 4, L, d_inner)
        x_dbl = self.x_proj(xs)  # (B, 4, L, dt_rank + 2*d_state)
        dt, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project dt
        dt = self.dt_proj(dt)  # (B, 4, L, d_inner)
        dt = dt.permute(0, 1, 3, 2)  # (B, 4, d_inner, L)

        # Prepare for selective scan
        xs = xs.permute(0, 1, 3, 2).reshape(B, -1, L)  # (B, 4*d_inner, L)
        dt = dt.reshape(B, -1, L)  # (B, 4*d_inner, L)
        Bs = Bs.reshape(B, K, -1, L)  # (B, 4, d_state, L)
        Cs = Cs.reshape(B, K, -1, L)  # (B, 4, d_state, L)

        # Fix A and D dimensions
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A = A.repeat(K, 1)  # (4*d_inner, d_state)
        D = self.D.float().repeat(K)  # (4*d_inner,)

        # Perform selective scan
        ys = selective_scan_fn(
            xs, dt, A, Bs, Cs, D,
            delta_softplus=True,
        ).view(B, K, -1, L)  # (B, 4, d_inner, L)

        # Combine scanning directions
        y = ys.sum(dim=1)  # (B, d_inner, L)
        y = y.transpose(1, 2).reshape(B, H, W, -1)  # (B, H, W, d_inner)

        # Output projection
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class MambaBlock(nn.Module):
    def __init__(self, dim, drop_path=0.2, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SS2D(d_model=dim, d_state=d_state)
        # 随机跳过一些网络模块或者子路径防止过拟合
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size

        # SSM branch
        x_2d = x.reshape(B, H, W, C)
        x_ssm = self.norm(x_2d)
        x_ssm = self.ssm(x_ssm)
        x_ssm = x_ssm.reshape(B, L, C)
        x = x + self.drop_path(x_ssm)

        # FFN branch
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = x + self.drop_path(x_ffn)

        return x


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

