'''
数据集读入
maziqing
2024/07/07
'''

import torch.utils.data as udata
from data_make1 import *

class Dataset(udata.Dataset):
    def __init__(self, img_path, L):
        super(Dataset, self).__init__()
        # self.respon_path = opt.respon_path
        # self.txt_path = txt_path
        self.img_path = img_path
        # self.patch_size = opt.patch_size
        # self.num_img = opt.num_img
        # self.sigma = opt.sigma
        # self.k = opt.k
        # self.ratio = opt.ratio
        # self.data = read_img(self.respon_path, self.txt_path, self.img_path, self.patch_size, 
        #     self.num_img, self.sigma, self.k, self.ratio)
        self.data = aug_img(img_path, 128, 32, L)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(len(self.data))
        hrhs = self.data[index][0]
        # hrhs = np.squeeze(hrhs, 0)
        # print(hrhs.shape)
        # print(torch.max(hrhs))
        hrms = self.data[index][1]
        # hrms = np.squeeze(hrms, 0)
        # print(torch.max(hrms))
        return hrhs, hrms
