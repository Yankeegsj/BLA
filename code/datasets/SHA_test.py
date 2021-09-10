import os
import sys
import numpy as np
from torch.utils import data
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import sparse
import scipy
import torch.nn.functional as F
import math
import cv2
from tools.progress.bar import Bar
import pickle
import json
import pdb
import torch.nn as nn
from torch.autograd import Variable
import math
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)
        
        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)
        # # Froze this Gaussian net
        if froze: self.frozePara()

    def forward(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps
    
    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False

class Gaussianlayer(nn.Module):
    def __init__(self, sigma=None, kernel_size=15):
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
            # sigma = [kernel_size * 0.3]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size // 2, froze=True)

    def forward(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps

class gen_den_map(nn.Module):
    def __init__(self,sigma,kernel_size):
        super(gen_den_map, self).__init__()
        self.gs = Gaussianlayer(sigma=[sigma],kernel_size=kernel_size)
    
    def forward(self, dot_map):
        # b*1*h*w
        gt_map = self.gs(dot_map)
        return gt_map

def sha_sort_img(filename):
    return int(filename.split('_')[1].split('.')[0])

class SHA_dataloader(data.Dataset):
    def __init__(self,split,opt):
        self.split=split
        self.opt=opt
        self.mean=opt.mean_std[0]
        self.std=opt.mean_std[1]

        self.gen_map=gen_den_map(opt.sigma, opt.kernel_size)

        self.img_root=os.path.join(opt.dataroot_SHA,'val','pre_load_img')

        imgfiles=[filename for filename in os.listdir(self.img_root) \
                       if os.path.isfile(os.path.join(self.img_root,filename)) and filename.split('.')[1] in ['jpg','png']]
        imgfiles.sort(key=sha_sort_img)

        with open(os.path.join(os.path.join(opt.dataroot_SHA,'{}_dot_pre_load_img.json'.format(self.split))),'r') as f:
            self.points_data=json.load(f)
        self.imgfiles=imgfiles
        self.gray_list=[]
        print('SHA test {}'.format(len(self.imgfiles)))

    def __getitem__(self,index):
        img_file=self.imgfiles[index]
        img_path=os.path.join(self.img_root,img_file)
        ann_point=self.points_data[img_file]
        gray=index in self.gray_list

        img=cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img=img.transpose([2,0,1])# c h w
        img=torch.from_numpy(img).float()
        img=img.div(255)
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        
        src_h,src_w=img.size(1),img.size(2)

        scale_factor=1.0

        h=int(src_h*scale_factor)
        h=h if h%self.opt.model_scale_factor==0 else h+self.opt.model_scale_factor-h%self.opt.model_scale_factor
        w=int(src_w*scale_factor)
        w=w if w%self.opt.model_scale_factor==0 else w+self.opt.model_scale_factor-w%self.opt.model_scale_factor
        size=(h,w)
        scale_factor_h=h/src_h
        scale_factor_w=w/src_w
        if scale_factor_h==1. and scale_factor_w==1.:
            pass
        else:
            img = F.interpolate(img.unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
            img = img.squeeze(0)

        den=torch.zeros(h,w)

        cnt=0
        for i in range(len(ann_point)):
            y=int((ann_point[i][0])*scale_factor_h)
            x=int((ann_point[i][1])*scale_factor_w)
            den[y,x]+=1.
            cnt+=1
        den=self.gen_map(den.unsqueeze(0).unsqueeze(0))
        den*=self.opt.gt_factor
        den=den.squeeze()

        info={'dataset':'SHA_test','img_file':img_file,'img_path':img_path,'gt_cnt':cnt}

        return img,den,info

    def __len__(self):
        return len(self.imgfiles)
        
def SHA_test(opt):

    data_test=SHA_dataloader('val',opt)

    test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
        worker_init_fn=np.random.seed(opt.seed))

    return test_loader,data_test

def data_collate(data):
    img,den,info= zip(*data)
    return img,den,info
