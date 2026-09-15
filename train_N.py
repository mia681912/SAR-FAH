# code for NPDE
# Author: ZiQing Ma
# 2024/09/10

import torch.nn as nn
from torch.utils import data
from model_paper import HaarSolve
# from readfromh5 import DatasetFromHdf5
from unet_high import *
from dataset_our1 import Dataset
import matplotlib
from compute import *
matplotlib.use('Agg')
import os
import glob
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms, utils
import numpy as np
from skimage.metrics import structural_similarity as ssim_cal1
from skimage.metrics import peak_signal_noise_ratio as psnr_cal1
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
# from argas_haar import args_n
from argas_tr import args_n
import cv2 as cv

opt = args_n()


# ================== Pre-Define =================== #
seed = np.random.RandomState(44999)
random_seed = 44999
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum

# 文件保存路径
if not os.path.isdir(opt.save_models):
    os.makedirs(opt.save_models)
model_folder = os.path.join(opt.save_models, 'model')
model_folder_best = os.path.join(opt.save_models, 'model_best')


argsDict = opt.__dict__
with open(os.path.join(opt.save_models, 'setting.txt'), 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')
    f.close()

# 加载参数
lr = opt.lr
epochs = opt.epochs
batch_size = opt.batch_size

# 加载模型
model = HaarSolve(opt).cuda()
# PLoss = nn.MSELoss().cuda()
PLoss = nn.L1Loss().cuda()


# 优化器
optimizer = optim.Adam(model.parameters(), lr = 1e-3,
                            weight_decay = 1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max=20, eta_min=1e-6, last_epoch=-1)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
#                                             step_size = 10,
#                                             gamma = 0.5)


# 保存模型
def save_checkpoint(model_folder1, model, epoch):
    model_out_path = os.path.join(model_folder1, '{}.pth'.format(epoch))

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }
    if not os.path.isdir(model_folder1):
        os.makedirs(model_folder1)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- Main Train ----------------------------------
###################################################################

def train(start_epoch=0, RESUME=False):
    if RESUME:
        path_checkpoint = model_folder + "{}.pth".format(opt.depoch)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))

    time_s = time.time()
    lr_now = 0
    psnr_best = 0
    # trainset = DatasetFromHdf5(opt.trainh5)
    trainset = Dataset(opt.trainh5, opt.L)
    trainload = data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_valid_loss = [], []
        psnr_train, psnr_valid = [], []
        ssim_train, ssim_valid = [], []
        lr_list = []

        # ============Epoch Train=============== #
        model.train()
        for batch_count, batch_data in enumerate(trainload):
            # 加载干净图像
            img_clean = batch_data[0].cuda()
            img_noise = batch_data[1].cuda()
            
            pred_image = model(img_noise.cuda())
            # print(torch.max(torch.max(pred_image)))
            # =========================计算loss========================================
            # loss = PLoss(pred_image.float(), img_clean.float()) 
            loss = PLoss(pred_image.float(), img_clean.float()) 
            epoch_train_loss.append(loss.item())
            # =========================参数更新=========================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            psnr_train.append((psnr_cal(torch.clamp(img_clean, 0, 255.), torch.clamp(pred_image, 0, 255.))).item())
            # ssim_train.append(compare_ssim(img_clean.cpu().detach().numpy().squeeze(0), pred_image.cpu().detach().numpy().squeeze(0), channel_axis=2))

            if (batch_count + 1) % 200 == 0:
                print("iter [{}]--[{}]/[{}]: train_loss is {}".format(epoch, batch_count, len(trainload), loss))

        lr_list.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()  # update lr

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_psnr = sum(psnr_train) / len(psnr_train)
        # train_ssim = sum(ssim_train) / len(ssim_train)

        # ============================可视化================================================

     
        save_checkpoint(model_folder, model, epoch)
        torch.set_grad_enabled(False)
        model.eval()
        average_test_psnr = test(epoch)
        torch.set_grad_enabled(True)


def test(epoch):
    # 结果保存路径
    save_path = opt.save_models
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_img_dir = os.path.join(save_path, 'visual')
    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)
    model_folder = os.path.join(opt.save_models, 'model')
    path_checkpoint = os.path.join(model_folder, '{}.pth'.format(epoch))
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model.eval()

    torch.set_grad_enabled(False)

    imgs_test = glob.glob(os.path.join(opt.test_data, '*.tif'))
    imgs_test.sort()
    # print(imgs_test)

    val_psnr = torch.zeros(len(imgs_test))
    val_ssim = torch.zeros(len(imgs_test))

    for idx, img_test in enumerate(imgs_test):
        print(idx)
        img_test = Image.open(img_test)
        img_test = img_test.convert('L')
        img_test = np.expand_dims(img_test, axis=0)
        img_test_np = np.array(img_test)
        # 加噪声
        # img_test_np = ((img_test_np+1.0) / 256.0)**2  
        noise = seed.gamma(size=img_test_np.shape, shape=opt.L, scale=1 / opt.L)
        img_noise = img_test_np * noise
        # print("干净图像的最大数值", np.max(img_test_np))

        img_test_tensor = torch.from_numpy(img_test_np)
        img_noise = torch.from_numpy(img_noise)
        img_clean = img_test_tensor.unsqueeze(0).cuda()
        img_noise = img_noise.unsqueeze(0).cuda()
        img_est = model(img_noise.float())
        # print(torch.max(img_clean))
        img_est = img_est.detach()
        # print(torch.squeeze(torch.squeeze(img_clean, dim=0),dim=0).shape)
        psnr_test = psnr_cal1(np.clip(torch.squeeze(torch.squeeze(img_clean, dim=0),dim=0).cpu().numpy(),0.,255.), np.clip(torch.squeeze(torch.squeeze(img_est, dim=0),dim=0).cpu().numpy(),0.,255.), data_range=255.)
        ssim_test = ssim_cal1(np.clip(torch.squeeze(torch.squeeze(img_clean, dim=0),dim=0).cpu().numpy(),0.,255.), np.clip(torch.squeeze(torch.squeeze(img_est, dim=0),dim=0).cpu().numpy(),0.,255.), data_range=255.)
        with open(os.path.join(save_path, 'index.txt'), 'a+') as fp:
            fp.write("image {}: psnr={:.4f} ssim= {:.4f}\n".format(idx, psnr_test, ssim_test))
        fp.close()

        img_est1 = torch.squeeze(img_est, dim=1)
        img_est1 = torch.squeeze(img_est1, dim=0)
        img_est1 = img_est1.cpu().detach().numpy()

        img_noise1 = torch.squeeze(img_noise, dim=1)
        img_noise1 = torch.squeeze(img_noise1, dim=0)
        img_noise1 = img_noise1.cpu().detach().numpy()

        print("image {}: psnr={:.7f} \n".format(idx, psnr_test))

        val_psnr[idx] = psnr_test
        val_ssim[idx] = ssim_test

        save_img_path = os.path.join(save_img_dir, '{}.jpg'.format(idx))
        cv.imwrite(save_img_path, img_est1)
        save_img_path1 = os.path.join(save_img_dir, '{}_noise.jpg'.format(idx))
        cv.imwrite(save_img_path1, img_noise1)
    psnr_average = torch.mean(val_psnr)
    ssim_average = torch.mean(val_ssim)
    with open(os.path.join(save_path, 'index.txt'), 'a+') as fp:
        fp.write("average psnr={:.4f} ssim= {:.4f}\n".format(psnr_average, ssim_average))
    fp.close()
    return psnr_average


###################################################################
# ------------------- Main Function  -------------------
###################################################################
if __name__ == "__main__":
    train()


