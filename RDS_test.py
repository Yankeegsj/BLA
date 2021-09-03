#coding=utf-8
import os
import sys
sys.path.append(os.path.abspath((__file__)))
from options_train import global_variables as opt_conf
from tools.terminal_log import create_log_file_terminal,save_opt,create_exp_dir
from tools.progress.bar import Bar
import tools.godblessdbg as godblessdbg
from tools.log import AverageMeter
from tools.progress.bar import Bar
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
import random
import glob
import utils
from utils import earlystop,lr_decay,AverageMeter_array,AverageMeter_array_mask
import scipy.io as io
import shutil
from matplotlib import pyplot as plt
import cv2
import json
import math
from torch.autograd import Variable
from vision import save_keypoints_img

def draw_loss_line(loss_list,index_map,save_dir):
    loss=np.array(loss_list)
    x=range(1,len(loss_list[0])+1)
    plt.figure()
    plt.xlabel("Epoch",fontsize=25) 
    plt.ylabel("MAE",fontsize=25)
    j=0
    for i in index_map:
        plt.plot(x,loss[j],label=i,markersize=10,linewidth=2.5) 
        j+=1
    plt.yticks(size = 22)
    plt.xticks(size = 22)
    plt.grid(ls='--')
    plt.legend(fontsize=17)
    plt.savefig(os.path.join(save_dir,'mae.pdf'),bbox_inches = 'tight')
    plt.figure() 
    plt.close('all')



def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    #float占用四个字节，所以会占用所有未使用内存
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True


def test(epoch,split,test_vision,model,test_loader,opt,log):
    with torch.no_grad():
        model.eval()
        results = {}
        batch_time = AverageMeter()
        data_time = AverageMeter()
        mae = AverageMeter()
        mse = AverageMeter()
        bar = Bar('Testing', max=len(test_loader))
        end = time.time()

        for step, data in enumerate(test_loader):
            data_time.update(time.time() - end)
            end = time.time()
            res=model.inference(data,0,epoch,0,step,test_vision,len(test_loader),0,log)
            batch_time.update(time.time() - end)
            end = time.time()
            str_plus=''
            for k,v in res.items():
                str_plus+=' | {key:}:{value:.4f}'.format(key=k,value=v)
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                batch=step + 1,
                size=len(test_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )+str_plus
            bar.next()

        bar.finish()
        
    return res


if __name__ == '__main__':
    print(sys.argv)
    option = dict([arg.split('=') for arg in sys.argv[1:]])
    print(option)
    opt=opt_conf(**option)
    # occumpy_mem(0)
    seed_torch(opt.seed)
    log_output=create_log_file_terminal(opt.log_txt_path)
    save_opt(opt,opt.log_txt_path)
    scripts_to_save=glob.glob('./code/*')
    create_exp_dir(opt.log_root_path,scripts_to_save)
    log_output.info(os.path.realpath(__file__))
    log_output.info(sys.argv)
    if not torch.cuda.is_available():
        log_output.info('no gpu device available')
        sys.exit(1)
    
    from importlib import import_module
    net=import_module('models.{}'.format(opt.model))
    exec('model=net.{}(opt)'.format(opt.model))
    model = model.cuda()

    dataloader=import_module('datasets.{}'.format(opt.dataset))
    exec('train_loader,train_data,_,_=dataloader.{}(opt)'.format(opt.dataset))
    if opt.dataset!='GCC_DR':
        train_data.mix_domain_set(opt.searched_domain)
    dataloader=import_module('datasets.{}'.format(opt.tar_dataset))
    exec('_,_,test_loader_tar,test_data_tar=dataloader.{}(opt)'.format(opt.tar_dataset))

    sample_num=min(len(test_loader_tar),opt.vision_each_epoch)
    test_vision_tar=random.sample(list(range(len(test_loader_tar))), sample_num)
    test_vision_tar.append(0)

    assert opt.model_for_load
    checkpoint=torch.load(opt.model_for_load)
    model.load_state_dict(checkpoint['net'])
    model.next_epoch()
    results=test(0,'test', test_vision_tar, model, test_loader_tar,opt,log_output)
    log_output.info('Results in {}'.format(opt.tar_dataset))
    for k,v in results.items():
        log_output.info('{}: {}'.format(k,v))
    log_output.info(godblessdbg.end)
    log_output.info('log location {0}'.format(opt.log_root_path))
