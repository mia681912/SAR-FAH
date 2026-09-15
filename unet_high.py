import torch
import torch.nn as nn
from argas_haar import args_n
from einops import rearrange, repeat
import os
import numpy as np
import torch.nn.functional as F
from mamba_block import MambaBlock as mamba
from mamba_block import PatchEmbed2D
from timm.models.layers import trunc_normal_
from defor_conv import Dynamic_conv2d as DCN


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
        in_channel / (r ** 2)), r * in_height, r * in_width
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



class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
 
        self.modulation = modulation # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)
 
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
 
    def forward(self, x):
        # print(x.shape)
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
 
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
 
        if self.padding:
            x = self.zero_padding(x)
 
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
 
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
 
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
 
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
 
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
 
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
 
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
 
        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
 
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
 
        return out
 
    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
 
        return p_n
 
    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
 
        return p_0
    
    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
 
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
 
    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
 
        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
 
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
 
        return x_offset
 
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
 
        return x_offset


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetConvBlock, self).__init__()
        # self.conv1 = DeformConv2d(in_channels, out_channels)
        # self.conv2 = DeformConv2d(out_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNetConvBlock1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetConvBlock1, self).__init__()
        # self.conv1 = DeformConv2d(in_channels, out_channels)
        self.conv2 = DeformConv2d(out_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetDown, self).__init__()
        self.conv_block = UNetConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv_block(x)
        x_pooled = self.pool(x_conv)
        return x_pooled, x_conv

class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetUp, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock1(out_channels, out_channels)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = x_skip + x
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, channels):
        super(UNet, self).__init__()

        self.down1 = UNetDown(channels, channels)
        self.down2 = UNetDown(channels, channels*2)

        # self.bottleneck = UNetConvBlock(channels*2, channels*2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 3, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
            DeformConv2d(channels*2, channels*2),
            # nn.Conv2d(channels*2, channels*2, 3, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU()
        )

        self.attention = mamba_denoise(channels*2, 2)
        self.att = CBEM(channels*2)
        self.de_8 = nn.Sequential(
            DCN(channels*4, channels*2, 1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU()
        )

        self.up3 = UNetUp(channels*2, channels*2)
        self.up4 = UNetUp(channels*2, channels)

        # self.final_conv = nn.Conv2d(128, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x1_conv = self.down1(x)
        # print(x1.shape)  #[64,10,10]
        x2, x2_conv = self.down2(x1)

        x_bottleneck = self.bottleneck(x2)
        # print(x_bottleneck.shape)

        x_bottleneck = self.de_8(torch.cat([self.attention(x_bottleneck), self.att(x_bottleneck)], 1))

        x_up3 = self.up3(x_bottleneck, x2_conv)
        x_up4 = self.up4(x_up3, x1_conv)

        out = x_up4
        return out


class mamba_denoise(nn.Module):
    def __init__(self, hidden_dim, patch_size):
        super(mamba_denoise, self).__init__()
        self.patch = patch_size
        self.ll_feature = hidden_dim
        self.hidden = hidden_dim
        # self.conv = PatchEmbed2D(patch_size, hidden_dim, self.hidden)
        self.self_attention = mamba(self.hidden, d_state=32)
        self.output = nn.Conv2d(self.hidden, hidden_dim, kernel_size=1)

        self.denoise = nn.Sequential(
            nn.Conv2d(self.ll_feature, self.hidden, 1),
            nn.ReLU()
        )
    # 位置嵌入
    def pos_embed(self, embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size, img_size)
        # print(patch_height)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def forward(self, x):
        # 将图像进行分块
        x = self.denoise(x)
        # x = self.conv(x)
        B, C, H, W = x.shape
        # print(x.shape)
        # print(self.pos_embed(C, self.patch, H).shape)
        x = x + self.pos_embed(C, self.patch, H).cuda()
        x_size = (H, W)
        x = x.flatten(2).transpose(1, 2)  # B, L, C
        # print(x.shape)
        x = self.self_attention(x, x_size)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.output(x)
        return x


class LOWDCN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LOWDCN, self).__init__()

        self.cbam = CBEM(in_feature)
        self.mamba = mamba_denoise(in_feature, 2)
        # self.dcn = DCN(2 * in_feature, out_feature, 3)
        self.fusion = nn.Sequential(
            DCN(2 * in_feature, out_feature, 1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.cbam(x)
        x2 = self.mamba(x)
        out = self.fusion(torch.cat([x1, x2], 1))

        return out


class LL_denoise(nn.Module):
    def __init__(self, ll_channel, ll_feature, denoise_num, device):
        super(LL_denoise, self).__init__()

        self.in_channel = ll_channel
        self.ll_feature = ll_feature
        self.ksize5 = 3
        self.ksize3 = 3
        self.ksize1 = 3

        layers = []
        layers.append(nn.Conv2d(self.in_channel+1, self.ll_feature, 1))
        for i in range(2):
            layers.append(nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize5, padding=1))
            layers.append(nn.BatchNorm2d(self.ll_feature))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize3, padding=1))
            layers.append(nn.BatchNorm2d(self.ll_feature))
            layers.append(nn.ReLU())
            layers.append(LOWDCN(self.ll_feature, self.ll_feature))
            layers.append(nn.Conv2d(self.ll_feature, self.ll_feature, 3, padding=1))
            layers.append(nn.BatchNorm2d(self.ll_feature))
            layers.append(nn.ReLU())
        self.denoise = nn.Sequential(*layers)
        self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, t, image_ll):
        input = self.concat_t(image_ll, t)
        out = self.denoise(input)

        return out

    def concat_t(self, h: torch.Tensor, t):
        h_shape = h.shape
        tt = torch.ones(h_shape[0], 1, h_shape[2], h_shape[3]).to(self.dev) * t
        out_ = torch.cat((h, tt), dim=1).float()  # shape =[bach_size, features+1, N_x, N_y]
        return out_


class Super_Attention(nn.Module):
    def __init__(self, ll_channel, ll_feature):
        super(Super_Attention, self).__init__()

        self.in_channel = ll_channel
        self.feature = ll_feature

        self.input = UNet(ll_channel//2)
        self.en_2 = nn.Sequential(
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature //2 , 3, padding=1),
            nn.BatchNorm2d(self.feature// 2),
            nn.ReLU()
        )
        self.en_4 = nn.Sequential(
            nn.Conv2d(self.feature//2, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU()
        )

    def forward(self, image_ll):
        out = self.en_4(self.input(self.en_2(image_ll)))

        return out
