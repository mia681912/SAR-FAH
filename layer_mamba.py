"""
去斑的小波变换以及一些基本的网络层
coder: Ziqing Ma  time:2024/09/22
"""

import torch
import torch.nn as nn
from argas_haar import args_n
from mamba_ssm import Mamba
from timm.models.layers import DropPath
import math
from einops import rearrange, repeat
import os
import numpy as np
import torch.nn.functional as F
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

opt = args_n()

#os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum

# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维离散小波逆变换
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBEM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBEM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SpaEmbe(nn.Module):
    def __init__(self, ms_channel, out_channel):
        super(SpaEmbe, self).__init__()
        k_size4 = 7
        k_size1 = 5
        k_size2 = 3
        k_size3 = 1
        # out_channel = 128
        self.conv7 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size4, padding=(k_size4 - 1) // 2, bias=False),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size1, padding=(k_size1 - 1) // 2, bias=False),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size2, padding=(k_size2 - 1) // 2, bias=False),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size3, bias=False),
            nn.ReLU()
        )

        self.conv = nn.Conv2d(4 * out_channel, out_channel, k_size3, padding=(k_size3 - 1) // 2, bias=False)

    def forward(self, lrhs):
        out7 = self.conv7(lrhs)
        out5 = self.conv5(lrhs)
        out3 = self.conv3(lrhs)
        out1 = self.conv1(lrhs)

        out = torch.cat([out7, out5, out3, out1], 1)
        out = self.conv(out)

        return out


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


from torchvision.ops import DeformConv2d


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=1, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output



class MultiScale(nn.Module):
    def __init__(self, inchan, outchan):
        super(MultiScale, self).__init__()
        d1 = 1
        d2 = 3
        d4 = 5
        self.c1 = nn.Sequential(
            nn.Conv2d(inchan, outchan, kernel_size=3, dilation=d1, padding=d1),
            # nn.BatchNorm2d(outchan),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(inchan, outchan, kernel_size=3, dilation=d2, padding=d2),
            # nn.BatchNorm2d(outchan),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(inchan, outchan, kernel_size=3, dilation=d4, padding=d4),
            # nn.BatchNorm2d(outchan),
            nn.ReLU()
        )

        self.fusion1 = nn.Conv2d(3 * outchan, outchan, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x4 = self.c4(x)

        out = self.fusion1(torch.cat([x1, x2, x4], 1))

        return out


class mamba_denoise(nn.Module):
    def __init__(self, hidden_dim, ):
        super(mamba_denoise, self).__init__()
        
        self.conv = nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1)
        self.self_attention = MambaBlock(hidden_dim  // 4, d_state=16)
        self.output = nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x_size = (H, W)
        x = x.flatten(2).transpose(1, 2)  # B, L, C
        x = self.self_attention(x, x_size)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.output(x)
        return x


class LL_denoise(nn.Module):
    def __init__(self, ll_channel, ll_feature, denoise_num, device):
        super(LL_denoise, self).__init__()

        self.in_channel = ll_channel
        self.ll_feature = ll_feature
        self.ksize5 = 3
        self.ksize3 = 3
        self.ksize1 = 3

        layers = []
        self.input = nn.Conv2d(self.in_channel+1, self.ll_feature, 1)
        self.denoise1 = nn.Sequential(
            nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize5, padding=1),
            nn.BatchNorm2d(self.ll_feature),
            nn.ReLU(),
            nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize3, padding=1),
            nn.BatchNorm2d(self.ll_feature),
            nn.ReLU()
        )
        self.mamba1 = mamba_denoise(self.ll_feature)

        self.df1 = Dynamic_conv2d(self.ll_feature, self.ll_feature, 3)

        self.denoise2 = nn.Sequential(
            nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize5, padding=1),
            nn.BatchNorm2d(self.ll_feature),
            nn.ReLU(),
            nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize3, padding=1),
            nn.BatchNorm2d(self.ll_feature),
            nn.ReLU()
        )

        self.att = CBEM(self.ll_feature)

        self.denoise3 = nn.Sequential(
            nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize5, padding=1),
            nn.BatchNorm2d(self.ll_feature),
            nn.ReLU(),
            nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize3, padding=1),
            nn.BatchNorm2d(self.ll_feature),
            nn.ReLU()
        )

        self.fusion1 = nn.Conv2d(self.ll_feature, self.ll_feature, 1)
    
        self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, t, image_ll):
        input = self.concat_t(image_ll, t)
        out = self.input(input)
        x1 = self.denoise1(out)
        # print(x1.shape)
        x_mamba = self.mamba1(out)
        # print(x_mamba.shape)
        x2 = self.df1(self.fusion1(x1 + x_mamba))
        # print(x2.shape)
        out = self.denoise3(self.att(self.denoise2(x2)))
        # print(out.shape)
        return out

    def concat_t(self, h: torch.Tensor, t):
        h_shape = h.shape
        tt = torch.ones(h_shape[0], 1, h_shape[2], h_shape[3]).to(self.dev) * t
        out_ = torch.cat((h, tt), dim=1).float()  # shape =[bach_size, features+1, N_x, N_y]
        return out_


# 无NODE的模块
class Super_Attention(nn.Module):
    def __init__(self, hh_channel, hh_feature):
        super(Super_Attention, self).__init__()

        self.in_channel = hh_channel
        self.feature = hh_feature

        self.att1 = CBEM(self.feature)

        self.df1 = Dynamic_conv2d(self.feature, self.feature, 3)
        self.ms1 = MultiScale(self.feature,self.feature)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU()
        )

        self.mam = mamba_denoise(self.feature)
        self.fusion1 = nn.Conv2d(2 * self.feature, self.feature, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU()
        )


    def forward(self, hh):
        out1 = self.conv1(hh)
        out2 = self.mam(hh)
        x2 = self.df1(self.fusion1(torch.cat([out1, out2], 1)))
        out = self.ms1(self.conv2(x2))
        out2 = out1 + self.att1(out1)
        out = self.conv3(out2)
        return out

