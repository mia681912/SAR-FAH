# """
# 计算指标
# 2024/09/11
# """


# import torch
# import torch.nn.functional as F
# from math import exp


# def gaussian(window_size, sigma) :
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()

# def create_window(window_size, channel) :
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def _ssim_map(img1, img2, window, window_size, channel) :
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = (0.01*255.) ** 2
#     C2 = (0.03*255.) ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     return ssim_map


# # def calc_ssim(img1, img2, k1=0.01, k2=0.03, mode=''):
# #         l = 255.
# #         info = img1.shape
# #         img1 = img1.reshape([-1, info[-1]])
# #         img2 = img2.reshape([-1, info[-1]])
# #         u1 = torch.mean(img1, axis=0).reshape([1, -1])
# #         u2 = torch.mean(img2, axis=0).reshape([1, -1])
# #         Sig1 = torch.std(img1, axis=0).reshape([1, -1])
# #         Sig2 = torch.std(img2, axis=0).reshape([1, -1])
# #         sig12 = torch.sum((img1 - u1) * (img2 - u2), axis=0) / (info[0] * info[1] - 1)
# #         c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
# #         print((2 * u1 * u2 + c1) * (2 * sig12 + c2))
# #         SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
# #         return torch.mean(SSIM)

# def calc_ssim(img1, img2, window_size = 11, size_average = True) :
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     ssim_map = _ssim_map(img1, img2, window, window_size, channel)

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

# def calc_psnr(img1, img2):
#     return 10. * torch.log10(255.**2 / torch.mean((img1 - img2) ** 2))


"""
计算 psnr, ssim, ergas, sam, uqi, dg, enl 等指标
GPU 版本
coder: Ziqing Ma  time:2024/09/12
"""

import torch
import numpy as np


# 输入为 [N, C, H, W]
# def rmse_cal(img_tgt, img_fus):
#     """

#     :param img_tgt: 维度为 [n,c,h,w]
#     :param img_fus: 维度为 [n,c,h,w]
#     :return: 均方误差
#     """

#     img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
#     img_fus = img_fus.reshape(img_fus.shape[0], -1)
#     rmse = np.sqrt(np.mean((img_tgt - img_fus)**2))

#     return rmse

def rmse_cal(img_tgt, img_fus):
    """

    :param img_tgt: 维度为 [n,c,h,w]
    :param img_fus: 维度为 [n,c,h,w]
    :return: 均方误差
    """

    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = (np.mean(abs(img_tgt - img_fus)))

    return rmse


def psnr_cal(img_tgt, img_fus):
    """

    :param img_tgt: 维度为 [n,c,h,w]
    :param img_fus: 维度为 [n,c,h,w]
    :return: 平均 PSNR
    """
    img_tgt = img_tgt.reshape([img_tgt.shape[0]*img_tgt.shape[1], -1])  # [nc,h*w]
    img_fus = img_fus.reshape([img_fus.shape[0]*img_fus.shape[1], -1])   # [nc, h*w]
    # img_tgt = img_tgt * 255.
    # img_fus = img_fus * 255.
    mse = torch.mean(torch.pow(img_tgt - img_fus, 2), axis=1)
    img_max = 255.
    psnr = 10 * torch.log10(img_max ** 2 / mse)
    # print(psnr)

    return torch.mean(psnr)


def sam_cal(img_tgt, img_fus):
    """

    :param img_tgt: 维度为 [n,c,h,w]
    :param img_fus: 维度为 [n,c,h,w]
    :return: 平均光谱角,以通道为基准计算偏角
    """

    img_tgt = img_tgt.reshape(img_tgt.shape[0]*img_tgt.shape[1], -1)     #[nc,h*w]
    img_fus = img_fus.reshape(img_fus.shape[0]*img_fus.shape[1], -1)

    A = torch.sqrt(torch.sum(img_tgt ** 2, axis=0))
    B = torch.sqrt(torch.sum(img_fus ** 2, axis=0))
    AB = torch.sum(img_tgt * img_fus, axis=0)

    sam = AB / torch.clamp((A * B), min=1e-6)
    sam = sam.clamp(-1, 1)
    sam = torch.arccos(sam)
    sam = torch.mean(sam) * 180 / torch.pi

    return sam


def ergas_cal(img_tgt, img_fus, scale):
    """

        :param img_tgt: 维度为 [n,c,h,w]
        :param img_fus: 维度为 [n,c,h,w]
        :return: 正则化全局误差
        """

    img_tgt = img_tgt.reshape(img_tgt.shape[0]*img_tgt.shape[1], -1)
    img_fus = img_fus.reshape(img_fus.shape[0]*img_fus.shape[1], -1)

    rmse = torch.mean((img_tgt-img_fus)**2, axis=1)        # 均方误差
    rmse = torch.sqrt(rmse)
    mean = torch.mean(img_tgt, axis=1)

    ergas = torch.mean((rmse/mean)**2)
    ergas = 100 / scale * ergas **0.5

    return ergas


def ssim_cal(x1, x2, max_v, k1=0.01, k2=0.03):
    """

        :param x1: 维度为 [n,c,h,w]
        :param x2: 维度为 [n,c,h,w]
        :return: 平均ssim
    """

    h, w = x1.shape[2], x1.shape[3]
    x1 = x1.reshape(x1.shape[0] * x1.shape[1], -1)
    x2 = x2.reshape(x2.shape[0] * x2.shape[1], -1)
    u1 = torch.mean(x1, axis=1).reshape([-1, 1])
    u2 = torch.mean(x2, axis=1).reshape([-1, 1])
    Sig1 = torch.std(x1, axis=1).reshape([-1, 1])
    Sig2 = torch.std(x2, axis=1).reshape([-1, 1])
    sig12 = torch.sum((x1 - u1) * (x2 - u2), axis=1) / (h * w - 1)
    sig12 = sig12.reshape([-1, 1])
    c1, c2 = pow(k1 * max_v, 2), pow(k2 * max_v, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))

    return torch.mean(SSIM)

