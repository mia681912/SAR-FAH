# 2025//06/04
# Ziqing Ma
# Mamba-based network for SAR-despeckling

from tkinter import N
import torch
import os
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
# from new_lay import *
from unet_high import *

device = torch.device('cuda:1')

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Res_bottle(nn.Module):
    def __init__(self, in_channel):
        super(Res_bottle, self).__init__()

        self.hs_feature = in_channel
        res_conv = nn.Sequential(
            nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
            # nn.BatchNorm2d(self.hs_feature),
            nn.ReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
            # nn.BatchNorm2d(self.hs_feature),
            nn.ReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
            # nn.BatchNorm2d(self.hs_feature),
            nn.ReLU()
        )
        self.res1 = Residual(res_conv)

    def forward(self, lrhs):
        out = self.res1(lrhs)

        return out


class De_solver(nn.Module):
    def __init__(self, t, N_t, solver, ll_channel, ll_feature, denoise_num, device, odeint_adjoint):
        super(De_solver, self).__init__()

        self.odeint_adjoint = odeint_adjoint
        self.t = t
        self.N_t = N_t
        self.solver = solver
        self.ODE_vector_field = LL_denoise(ll_channel, ll_feature, denoise_num, device)
        self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, noise_image: torch.Tensor):
        if self.odeint_adjoint:
            print('this')
            odeint_solver = odeint_adjoint
        else:
            odeint_solver = odeint
        # 生成时间空间  N_t是个数
        # print(self.odeint_adjoint)
        timesteps = torch.from_numpy(np.linspace(0, self.t, self.N_t + 1)).to(self.dev)
        out = odeint_solver(func=self.ODE_vector_field, y0=noise_image, t=timesteps, method=self.solver)[-1]

        return out


class SR_solver(nn.Module):
    def __init__(self, t, N_t, solver, hh_channel, hh_feature, device, odeint_adjoint):
        super(SR_solver, self).__init__()

        self.odeint_adjoint = odeint_adjoint
        self.t = t
        self.N_t = N_t
        self.solver = solver
        self.ODE_vector_field = Super_Attention(hh_channel, hh_feature, device)

        self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, noise_image: torch.Tensor):
        if self.odeint_adjoint:
            odeint_solver = odeint_adjoint
        else:
            odeint_solver = odeint
        # 生成时间空间  N_t是个数
        timesteps = torch.from_numpy(np.linspace(0, self.t, self.N_t + 1)).to(self.dev)
        out = odeint_solver(func=self.ODE_vector_field, y0=noise_image, t=timesteps, method=self.solver)[-1]

        return out


class HaarSolve(nn.Module):
    def __init__(self, args):
        super(HaarSolve, self).__init__()

        # self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        self.in_channel = args.num_channel
        self.out_feature = args.out_channel

        self.img_dwt = DWT()
        self.img_iwt = IWT()

        self.conv_ll = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1),
            # nn.BatchNorm2d(self.out_feature),
            # nn.ReLU()
        ) 
        self.conv_lh = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1),
            # nn.BatchNorm2d(self.out_feature),
            # nn.ReLU()
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1),
            # nn.BatchNorm2d(self.out_feature),
            # nn.ReLU()
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1),
            # nn.BatchNorm2d(self.out_feature),
            # nn.ReLU()
        )

        # self.l_h1 = low_high(self.out_feature, self.out_feature)
        # self.l_h2 = low_high(self.out_feature, self.out_feature)
        # self.l_h3 = low_high(self.out_feature, self.out_feature)
        # self.h_l1 = high_low(self.out_feature, self.out_feature)
        # self.h_l2 = high_low(self.out_feature, self.out_feature)
        # self.h_l3 = high_low(self.out_feature, self.out_feature)

        self.ll_de = De_solver(args.t, args.N_t, args.solver, self.out_feature,
                               self.out_feature, args.denoise_num, args.device, args.odeint_adjoint)  # 三层的网络
        self.lh_sr1 = Super_Attention(self.out_feature, self.out_feature)
        self.lh_sr2 = Super_Attention(self.out_feature, self.out_feature)
        self.lh_sr3 = Super_Attention(self.out_feature, self.out_feature)
        self.fusion1 = nn.Sequential(
            nn.Conv2d(int(4 * self.out_feature), self.out_feature, 1),
            # nn.BatchNorm2d(self.out_feature),
            # nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(self.out_feature, 4, 1),
            # nn.BatchNorm2d(4),
            # nn.ReLU()
        )
        # self.fusion_ll = nn.Conv2d(int(3 * self.out_feature), self.out_feature, 1)

        self.renew = Res_bottle(self.out_feature)

    def forward(self, x):
        x_dwt = self.img_dwt(x)
        input_ll = self.conv_ll(x_dwt[:, 0:1, :, :])  # 单独处理，因为含有很多特征
        input_lh = self.conv_lh(x_dwt[:, 1:2, :, :])
        input_hl = self.conv_hl(x_dwt[:, 2:3, :, :])
        input_hh = self.conv_hh(x_dwt[:, 3:4, :, :])

        # input_ll_1 = self.l_h1(input_ll, input_lh)
        # input_ll_2 = self.l_h2(input_ll, input_lh)
        # input_ll_3 = self.l_h3(input_ll, input_lh)

        # input_lh = self.h_l1(input_ll, input_lh)
        # input_hl = self.h_l2(input_ll, input_hl)
        # input_hh = self.h_l3(input_ll, input_hh)

        # input_ll = self.fusion_ll(torch.cat([input_ll_1, input_ll_2, input_ll_3], 1))
        x_de = self.ll_de(input_ll)
        x_lh = self.lh_sr1(input_lh)
        x_hl = self.lh_sr2(input_hl)
        x_hh = self.lh_sr3(input_hh)

        x_cat = torch.cat([x_de, x_lh, x_hl, x_hh], dim=1)
        x_cat = self.fusion1(x_cat)
        out = self.renew(x_cat)
        out = self.fusion2(out)
        # x_ll, x_lh, x_hl, x_hh = torch.chunk(out, 4, dim=1)

        return self.img_iwt(out)
