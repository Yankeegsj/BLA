import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import random
import torch.backends.cudnn as cudnn
import os
from tools.log import AverageMeter
from vision import save_keypoints_img,save_img_tensor,save_keypoints_and_img
import pdb
import time

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn)
        self.bn = nn.BatchNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class den_level_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=512, num_classes=1,opt=None):
        super(den_level_Discriminator, self).__init__()
        self.grl=GradientScalarLayer(opt.gradient_scalar)

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        x=self.grl(x)
        out = self.D(x)
        out = self.cls(out)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out



class VGG_16_deconv_Fine_grained_DA_test(nn.Module):
    def __init__(self, opt):
        super(VGG_16_deconv_Fine_grained_DA_test, self).__init__()
        self.gradient_scalar=opt.gradient_scalar
        self.opt=opt
        self.step_epoch=0

        vgg = models.vgg16(pretrained=False)

        features = list(vgg.features.children())


        self.backbone = nn.Sequential(*features[0:23])

        self.den_pred = nn.Sequential(
                                     Conv2d(512, 128, 3, same_padding=True, bn=True, NL='relu'),
                                     BasicDeconv(128, 128, 2, 2, True),
                                     Conv2d(128, 64, 3, same_padding=True, bn=True, NL='relu'),
                                     BasicDeconv(64, 64, 2, 2, True),
                                     Conv2d(64, 32, 3, same_padding=True, bn=True, NL='relu'),
                                     BasicDeconv(32, 32, 2, 2, True),
                                     Conv2d(32, 1, 1, same_padding=True, NL=''),
                                     nn.ReLU(inplace=True),
                                     )
        self.domain_cls=den_level_Discriminator(512,512,len(self.opt.split_den_level)-1,opt)

    def next_epoch(self):
        self.test_reset()

    def test_reset(self):
        self.mae=AverageMeter()
        self.mse=AverageMeter()
        if 'WE' in self.opt.tar_dataset:
            self.mae1=AverageMeter()
            self.mae2=AverageMeter()
            self.mae3=AverageMeter()
            self.mae4=AverageMeter()
            self.mae5=AverageMeter()

    def get_test_res(self):
        res={}
        res['mae']=self.mae.avg
        res['mse']=self.mse.avg**0.5

        if 'WE' in self.opt.tar_dataset:
            res['mae1']=self.mae1.avg
            res['mae2']=self.mae2.avg
            res['mae3']=self.mae3.avg
            res['mae4']=self.mae4.avg
            res['mae5']=self.mae5.avg

        return res


    def inference(self,data,search_time,epoch,global_step,step,vision_list,batch_num_each_epoch,domain,logger):
        batch=len(data[0])
        for i in range(batch):
            c,h,w=data[0][i].size()
            x=self.backbone(data[0][i].unsqueeze(0).cuda())
            den_map=self.den_pred(x)
            sum_map=torch.sum(den_map[i].detach().cpu())/self.opt.gt_factor
          
            self.mae.update(abs(sum_map-data[2][i]['gt_cnt']))
            self.mse.update((abs(sum_map-data[2][i]['gt_cnt'])**2))

            if step in vision_list and i==0:
                video_name=os.path.basename(data[2][i]['dataset'])
                img_name=data[2][i]['img_file']

                save_root=os.path.join(self.opt.log_root_path,'test',str(epoch),data[2][i]['dataset'])

                save_filename=os.path.join(save_root,video_name+'gt_point_',img_name.split('.')[0]+'_cnt_'+str(data[2][i]['gt_cnt'])+'.'+img_name.split('.')[1])
                save_keypoints_and_img(data[0][i],data[1][i], self.opt, save_filename,img_name)
                save_filename=os.path.join(save_root,video_name+'out_point_',img_name.split('.')[0]+'_cnt_'+str(sum_map)+'.'+img_name.split('.')[1])
                save_keypoints_and_img(data[0][i],den_map[0,0].detach().cpu(), self.opt, save_filename,img_name)

        res=self.get_test_res()
        return res


