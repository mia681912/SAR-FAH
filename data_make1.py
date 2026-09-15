'''
 UCMerced_LandUse  dataset
读取数据集并制作训练集
训练集：21*70, 39690
测试集：21*15
验证集：21*1  8505
比例：7：1.5：0.1
写入 h5.py
maziqing
2024/11/20
'''
import random
import shutil

import cv2
import scipy.io as sio
import os
import pandas as pd
import numpy as np
import torch
import h5py


seed = np.random.RandomState(44999)

def from_num_tensor(data):
    # 转变为tensor变量
    data = np.transpose(data, (2, 0, 1))
    data = np.float32(data)
    # data = torch.from_numpy(data).permute(2, 0, 1)
    # 增加维度
    # data = torch.unsqueeze(data, dim=0)
    # data = data.float()
    return data

def data_augmentation(image, mode):
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out


def aug_ment_per_img(img, patch_size, stride, L):
    """

    :param img: 干净图像
    :param patch_size: 切块大小
    :param stride: 采样间隔
    :param L: 斑点噪声水平
    :param path_h5: 输出h5文件地址
    :return: 干净图 噪声图
    """

    aug_img = []
    w, h = img.shape[0], img.shape[1]  #img是灰度图，大小为[256*256*1]
    num = 0
    for i in range((w - stride) // patch_size + 1):
        for j in range((h - stride) // patch_size + 1):
        #任意切片的随机位置
            aug_w = i * stride
            aug_h = j * stride

            img_aug_per = img[aug_w:aug_w+patch_size, aug_h:aug_h+patch_size, :]
            
            noise = seed.gamma(size=img_aug_per.shape, shape=L, scale=1/L)
            noise_aug_per = img_aug_per * noise
            
            img_aug_per1 = from_num_tensor(img_aug_per)
            noise_aug_per1 = from_num_tensor(noise_aug_per)
            # print('noise', np.max(np.max(noise)))
            # print('clean', torch.max(torch.max(img_aug_per1)))
            # print(torch.max(torch.max(noise_aug_per1)))

            aug_img.append([img_aug_per1, noise_aug_per1])
            num += 1
            for m in range(4):
                img_aug_per11 = img_aug_per.copy()    # *1
                data_aug1 = data_augmentation(img_aug_per11, m)
                noise1 = seed.gamma(size=data_aug1.shape, shape=L, scale=1/L)
                # print('data_aug', np.max(np.max(data_aug1)))
                # data_aug1 = ((np.float32(data_aug1) + 1.0) / 256.0) ** 2
                noise_aug_per1 = data_aug1 * noise1
                data_aug1 = np.ascontiguousarray(data_aug1)  # *2 解决内存不连续，无法转换为tensor的问题
                noise_aug_per1 = np.ascontiguousarray(noise_aug_per1)

                img_aug_per2 = from_num_tensor(data_aug1)
                noise_aug_per2 = from_num_tensor(noise_aug_per1)
                # print('aug noise', torch.max(torch.max(noise_aug_per2)))
                aug_img.append([img_aug_per2, noise_aug_per2])
                num += 1
    # print(num)

    return aug_img


def shutil_img(img_path, new_file):
    """

    :param img_path: 原始数据集地址
    :param new_file: 新数据集地址
    :return: 0
    """

    #====================移动所有图片置于一个文件夹=====================================
    i = 0
    files = os.listdir(img_path)
    if not os.path.exists(new_file):
        os.mkdir(new_file)
    for file in files:
        file_path = os.path.join(img_path, file)
        all_img = os.listdir(file_path)
        for img in all_img:
            i+=1
            old_img = os.path.join(file_path, img)
            new_img = os.path.join(new_file, img)
            shutil.copy(old_img, new_img)
            # print(i)



# img_path = r'/home/mzq/file_soft/data/SAR_model/train'
# new_file = r'/home/mzq/file_soft/data/Model_data/L_10'
# patch_size = 64
# L = 196
# stride = 2
# path_h5 = ''
# dataset = shutil_img(img_path, new_file, patch_size, stride, L, path_h5)


# def aug_img(img_path, patch_size, stride, L, path_h5, num):
#     """

#     :param img_path: 数据集地址
#     :param patch_size: 块大小
#     :param stride: 间隔
#     :param L: 噪声水平
#     :param path_h5: 输出h5文件地址
#     :return:
#     """
#     i = 0
#     num_L = 0
#     train_num = 0
#     img_L = L
#     all_img = os.listdir(img_path)
#     h5f = h5py.File(path_h5, 'w')
#     clean = h5f.create_group('clean')
#     noise = h5f.create_group('noise')
#     i = 0
#     for img_per in all_img:
#         img_path_per = os.path.join(img_path, img_per)
#         img = cv2.imread(img_path_per, 0)
#         # print(img.shape)
#         img = np.expand_dims(img, 2)
#         i = i + 1
#         # single polar operator
#         # img = ((np.float32(img) + 1.0) / 256.0) ** 2

#         aug_img = aug_ment_per_img(img, patch_size, stride, img_L)
#         for n in range(len(aug_img)):
#             data = aug_img[n].copy()
#             clean1 = data[0]
#             noise1 = data[1]
#             # print(torch.max(torch.max(noise1)))
#             clean.create_dataset(str(train_num), data=clean1)
#             noise.create_dataset(str(train_num), data=noise1)
#             train_num += 1
            
#     h5f.close()
#     print(train_num)

# def aug_img(img_path, patch_size, stride, L):
#     """

#     :param img_path: 数据集地址
#     :param patch_size: 块大小
#     :param stride: 间隔
#     :param L: 噪声水平
#     :param path_h5: 输出h5文件地址
#     :return:
#     """
#     i = 0
#     num_L = 0
#     train_num = 0
#     img_L = L
#     all_img = os.listdir(img_path)
#     aug_img1 = []
    
#     # h5f = h5py.File(path_h5, 'w')
#     # clean = h5f.create_group('clean')
#     # noise = h5f.create_group('noise')
#     i = 0
#     for img_per in all_img:
#         img_path_per = os.path.join(img_path, img_per)
#         img = cv2.imread(img_path_per, 0)
#         # print(img.shape)
#         img = np.expand_dims(img, 2)
#         i = i + 1
#         # single polar operator
#         # img = ((np.float32(img) + 1.0) / 256.0) ** 2

#         aug_img = aug_ment_per_img(img, patch_size, stride, img_L)
#         for n in range(len(aug_img)):
#             data = aug_img[n].copy()
#             clean1 = data[0]
#             noise1 = data[1]
#             # print(torch.max(torch.max(noise1)))
#             aug_img1.append([clean1, noise1])
#             # train_num += 1
            
#     return aug_img1

def aug_img(img_path, patch_size, stride, L):
    """

    :param img_path: 数据集地址
    :param patch_size: 块大小
    :param stride: 间隔
    :param L: 噪声水平
    :param path_h5: 输出h5文件地址
    :return:
    """
    i = 0
    num_L = 0
    train_num = 0
    img_L = L
    all_img = os.listdir(img_path)
    aug_img1 = []
    for img_per in all_img:
    # for i in range(2):
        img_path_per = os.path.join(img_path, img_per)
        img = cv2.imread(img_path_per, 0)
        img = np.expand_dims(img, 2)

        aug_img = aug_ment_per_img(img, patch_size, stride, L)
        for n in range(len(aug_img)):
            data = aug_img[n].copy()
            clean1 = data[0]
            noise1 = data[1]
            aug_img1.append([clean1, noise1])
    return aug_img1



# img_path = r'/home/mzq/file_soft/data/SAR_model/train'
# patch_size = 40
# stride = 10
# num = 1000000
# # 共21*70 张图片
# L = 1.
# # L = [4, 4, 4, 4, 4]
# # L = [10, 10, 10, 10, 10]
# path_h5_file = r'/home/mzq/file_soft/data/new/L_1'
# if not os.path.exists(path_h5_file):
#     os.makedirs(path_h5_file)
# path_h5 = os.path.join(path_h5_file, 'train.h5')
# aug_img(img_path, patch_size, stride, L, path_h5, num)


def addnoise_img(img_path, new_file, L):
    """

    :param img_path: 原始数据集地址
    :param new_file: 新数据集地址
    :return: 0
    """

    #====================移动所有图片置于一个文件夹=====================================
    all_img = os.listdir(img_path)
    if not os.path.exists(new_file):
        os.makedirs(new_file)
    
    for img in all_img:
        old_img = os.path.join(img_path, img)
        new_img = os.path.join(new_file, img)
        img_data = cv2.imread(old_img, 0)
        img_data = np.expand_dims(img_data, 2)
        noise = np.random.gamma(L, 1 / L, img_data.shape)
        noise_data = img_data * noise
        # print(noise_data.shape)
        cv2.imwrite(new_img, np.uint8(noise_data))

# img_path = r'./data/train'
# new_file = r'./noise_data/L_1/train'
# L = 1
# addnoise_img(img_path, new_file, L)


def shutil_img(img_path, new_file1, new_file2, new_file3):
    """

    :param img_path: 原始数据集地址
    :param new_file: 新数据集地址
    :return: 0
    """

    #====================移动所有图片置于一个文件夹=====================================
    files = os.listdir(img_path)
    if not os.path.exists(new_file1):
        os.makedirs(new_file1)
    if not os.path.exists(new_file2):
        os.makedirs(new_file2)
    if not os.path.exists(new_file3):
        os.makedirs(new_file3)
    for file in files:
        i = 0
        file_path = os.path.join(img_path, file)
        all_img = os.listdir(file_path)
        random.shuffle(all_img)
        for img in all_img:
            old_img = os.path.join(file_path, img)
            img_data = cv2.imread(old_img, 0)
            if img_data.shape[0] == img_data.shape[1] and img_data.shape[0]==256:
                if i < 70:
                    new_img = os.path.join(new_file1, img)
                    shutil.copy(old_img, new_img)
                    i += 1
                elif i < 70+15:
                    new_img = os.path.join(new_file2, img)
                    shutil.copy(old_img, new_img)
                    i += 1
                elif i < 70+16:
                    new_img = os.path.join(new_file3, img)
                    shutil.copy(old_img, new_img)
                    i += 1


img_path = r'/home/mzq/file_soft/data/SAR/UCMerced_LandUse/Images'
new_file1 = r'/home/mzq/file_soft/data/SAR_model/train'
new_file2 = r'/home/mzq/file_soft/data/SAR_model/valid'
new_file3 = r'/home/mzq/file_soft/data/SAR_model/test'
# shutil_img(img_path, new_file1, new_file2, new_file3)
# all_train = os.listdir(new_file1)
# all_test = os.listdir(new_file3)
# all_val = os.listdir(new_file2)
# print(len(all_train))
# print(len(all_val))
# print(len(all_test))


