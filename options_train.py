# -*- coding: utf-8
import time
import os
import torch
import sys
import cv2
import warnings
class global_variables(object):

    num_gpu=1
    local_rank=0
    iter_each_train_epoch=0#训练时按照tar图片数量设置epoch
    # iter_each_test_epoch=0
    #global variables
    seed=1234
    
    time_=time.localtime()
    time_=time.strftime('%Y-%m-%d-%H-%M',time_)
    log_root_path=''
    log_txt_path=''
    
    '''
    data process
    '''
    comment=''
    dataset='SHA'
    # dataroot_SHA='/gongshenjian/counting/YZL_cvpr2021/codes/data/shanghaitechA'
    # dataroot_SHB='/gongshenjian/counting/YZL_cvpr2021/codes/data/shanghaitech'
    dataroot_QNRF='/gongshenjian/dataset/QNRF'
    dataroot_QNRF_pre_load='/gongshenjian/counting/YZL_cvpr2021/codes/data/UCF_QNRF/processed' 
    # dataroot_venice='/gongshenjian/dataset/venice/venice'
    dataroot_mall='/gongshenjian/dataset/MALL/Mall'
    dataroot_fdst='/gongshenjian/dataset/FDST'
    # dataroot_GCC='/gongshenjian/dataset/GCC_dataset/GCC'
    # dataroot_GCC2SHA='/gongshenjian/counting/unsupervisied/13_generate_da_imgs/PCEDA-master/PCEDA_Phase/results/gcc2sha'
    dataroot_WE_blurred='/gongshenjian/dataset/WorldExpo/World_Expo_blurred'
    dataroot_WE='/gongshenjian/dataset/WorldExpo'
    dataroot_UCF50='/gongshenjian/counting/YZL_cvpr2021/codes/data/UCF_CC_50'


    dataroot_GCC='/opt/data/common/datasets/GCC'
    dataroot_GCC_anno='/gongshenjian/dataset/GCC_dataset/GCC'
    dataroot_GCC2SHA='/opt/data/common/datasets/GCC/gcc2sha'
    dataroot_SHA='/opt/data/common/datasets/shanghaitechA'
    dataroot_SHB='/opt/data/common/datasets/shanghaitechB'
    dataroot_venice='/opt/data/common/datasets/venice'


    tar_dataset='SHA'
    UCF50_val_choose=1
    search=False
    train_crop_size=[0,0]
    tar_scale_factor=[1.0]
    tar_fix_scale_factor=1.0

    # only for GCC
    src_train_crop_size=[0,0]
    split_method='random'# location camera
    split_txt_path='/gongshenjian/dataset/GCC_dataset/description_and_split'
    level_regularization=[0,1,2,3,4,5,6,7,8]
    time_regularization=[0,25]#左闭右开
    weather_regularization=[0,1,2,3,4,5,6]
    count_range=[0,100000]#闭区间
    radio_range=[0.,100000.]#闭区间
    level_capacity=[10.,25.,50.,100.,300.,600.,1000.,2000.,4000.]

    # Learning from Synthetic Data for Crowd Counting in the Wild
    # Target Dataset level time weather count range ratio range
    # SHT A 4,5,6,7,8 6:00∼19:59 0,1,3,5,6 25∼4000 0.5∼1
    # SHT B 1,2,3,4,5 6:00∼19:59 0,1,5,6 10∼600 0.3∼1
    # UCF CC 50 5,6,7,8 8:00∼17:59 0,1,5,6 400∼4000 0.6∼1
    # UCF-QNRF 4,5,6,7,8 5:00∼20:59 0,1,5,6 400∼4000 0.6∼1
    # WorldExpo’10 2,3,4,5,6 6:00∼18:59 0,1,5,6 0∼1000 0∼1


    #search space
    cutmix_grid=[8,8]
    search_space=['grey','scale','FFT','cutmix','perspective_transform']
    # attributes_range={
    # 'grey':[0.,1.],
    # 'scale':[0.,1.],#default=1.
    # 'FFT':[0.,1.],
    # 'cutmix':[0.,1.],
    # 'perspective_transform':[0.,45.]
    # }
    scale_range_max=1.0
    attributes_range={
    'grey':[0.,1.],
    'scale':[0.,1.],#default=1.
    'FFT':[-4.,-1],#0.0001 0.1对数取法
    'cutmix':[0.,1.],
    'perspective_transform':[0.,45.]
    }
    searched_domain=[0,0,0,0,0]
    

    img_interpolate_mode='bilinear'# 'nearest' 'bicubic'
    mean_std=([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    HorizontallyFlip=True
    VerticallyFlip=False
    gt_factor=100.0
    sigma=4
    kernel_size=15

    num_workers=8
    train_batch_size   = 4
    test_batch_size   = 1

    #model
    model=''
    # model_pre_train=''
    model_resume_path=''
    model_for_load=''
    model_scale_factor=8# CSRNet 8  Res50 8 Hrnet  32  模型featuremap 最小分辨率与输入大小之间的比例
    model_interpolate_mode='bilinear'
    gradient_scalar=-1.0
    MLP_layer=[384,256,128]

    ###############################################################

    grid=[8,8]
    split_den_level=[0,10000]
    gt_den_level=False
    soft=False
    # loss
    loss_choose='loss1'#直接以这个为类名称调用选择的loss
    ####加BF_cls时起作用
    BF_loss_choose='BCE'#softmax
    BF_cls_weight=1.0
    BF_mask_den_loss_weight=0.
    BF_grid=[8,8]
    BF_thd=[0.005,10000]#在这个范围内的是前景 mask为1
    BF_kernel=1

    ####加map_refine
    refine_loss_weight=1.0


    # vision
    save_start_epochs=3
    vision_each_epoch=15
    vision_frequency=30

    #training set
    frozen_layers=['pass']
    secondary_layers=['pass']

    main_lr_init = 1e-5
    main_weight_decay=1e-4

    secondary_lr_init = 1e-5
    secondary_weight_decay=1e-4

    optimizer='Adam'
    train_mode='step'
    num_epochs   = 100
    decay_iter_freq=100
    decay_gamma=0.95
    # 0.95**100=0.006
    # 0.94**100=0.002
    # 0.93**100=7.051e-4
    # 0.92**100=2.392e-4
    # 0.91**100=8.019e-5
    # 0.90**100=2.656e-5

    def __init__(self, **kwself):
        for k, v in kwself.items():
            # print(k)
            # print('\n')
            if k=='--local_rank':
                k='local_rank'
            if not hasattr(self, k):
                print("Warning: opt has not attribut {}".format(k))
                import pdb
                pdb.set_trace()
                self.__dict__.update({k: v})
            tp = eval('type(self.{0})'.format(k))
            if tp == type(''):
                setattr(self, k, tp(v))
            elif tp == type([]):
                tp=eval('type(self.{0}[0])'.format(k))
                if tp==type('1'):
                    v=v[1:-1].split(',')
                    setattr(self, k, v)
                else:
                    setattr(self, k, eval(v))
            else:
                setattr(self, k, eval(v))

        if self.comment:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}_{}'.format(self.time_,self.model,self.comment))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}_{}.txt'.format(self.time_,self.model,self.comment))
        else:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}'.format(self.time_,self.model))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}.txt'.format(self.time_,self.model))
        
        if self.num_gpu>1:
            self.log_root_path+='_'+str(self.local_rank)
            self.log_txt_path=self.log_txt_path.split('.txt')[0]+'_'+str(self.local_rank)+'.txt'

        if 'pass' in self.frozen_layers:
            self.frozen_layers=[]

        if 'pass' in self.secondary_layers:
            self.secondary_layers=[]

        # if 'pass' in self.per_transform:
        #     self.per_transform=[]
        
        if 'GCC' not in self.dataset:
            self.split_method=None
            self.split_txt_path=None
            self.level_regularization=None
            self.time_regularization=None
            self.weather_regularization=None
            self.count_range=None
            self.radio_range=None
            self.level_capacity=None

        self.cutmix_grid=self.grid[0]
        self.attributes_range['scale']=[max(self.src_train_crop_size[0]/1080,self.src_train_crop_size[1]/1920),self.scale_range_max]

        assert self.test_batch_size==1
        if self.num_gpu>1:
            if not os.path.exists(self.log_root_path) and self.local_rank==0:
                os.makedirs(self.log_root_path)
        elif not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)

if __name__ == '__main__':
    option = dict([arg.split('=') for arg in sys.argv[1:]])
    opt = global_variables(**option)
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(opt, k))

    '''
    k_size=15
>>> sigma=4
>>> H = np.multiply(cv2.getGaussianKernel(k_size, sigma), (cv2.getGaussianKernel(k_size, sigma)).T)
>>> H
array([[0.00052666, 0.00079061, 0.00111494, 0.00147705, 0.00183822,
        0.0021491 , 0.00236032, 0.00243525, 0.00236032, 0.0021491 ,
        0.00183822, 0.00147705, 0.00111494, 0.00079061, 0.00052666],
       [0.00079061, 0.00118685, 0.00167372, 0.00221732, 0.0027595 ,
        0.00322618, 0.00354327, 0.00365574, 0.00354327, 0.00322618,
        0.0027595 , 0.00221732, 0.00167372, 0.00118685, 0.00079061],
       [0.00111494, 0.00167372, 0.00236032, 0.00312692, 0.00389152,
        0.00454964, 0.00499681, 0.00515542, 0.00499681, 0.00454964,
        0.00389152, 0.00312692, 0.00236032, 0.00167372, 0.00111494],
       [0.00147705, 0.00221732, 0.00312692, 0.0041425 , 0.00515542,
        0.0060273 , 0.00661969, 0.00682983, 0.00661969, 0.0060273 ,
        0.00515542, 0.0041425 , 0.00312692, 0.00221732, 0.00147705],
       [0.00183822, 0.0027595 , 0.00389152, 0.00515542, 0.00641603,
        0.0075011 , 0.00823834, 0.00849985, 0.00823834, 0.0075011 ,
        0.00641603, 0.00515542, 0.00389152, 0.0027595 , 0.00183822],
       [0.0021491 , 0.00322618, 0.00454964, 0.0060273 , 0.0075011 ,
        0.00876967, 0.0096316 , 0.00993734, 0.0096316 , 0.00876967,
        0.0075011 , 0.0060273 , 0.00454964, 0.00322618, 0.0021491 ],
       [0.00236032, 0.00354327, 0.00499681, 0.00661969, 0.00823834,
        0.0096316 , 0.01057824, 0.01091403, 0.01057824, 0.0096316 ,
        0.00823834, 0.00661969, 0.00499681, 0.00354327, 0.00236032],
       [0.00243525, 0.00365574, 0.00515542, 0.00682983, 0.00849985,
        0.00993734, 0.01091403, 0.01126048, 0.01091403, 0.00993734,
        0.00849985, 0.00682983, 0.00515542, 0.00365574, 0.00243525],
       [0.00236032, 0.00354327, 0.00499681, 0.00661969, 0.00823834,
        0.0096316 , 0.01057824, 0.01091403, 0.01057824, 0.0096316 ,
        0.00823834, 0.00661969, 0.00499681, 0.00354327, 0.00236032],
       [0.0021491 , 0.00322618, 0.00454964, 0.0060273 , 0.0075011 ,
        0.00876967, 0.0096316 , 0.00993734, 0.0096316 , 0.00876967,
        0.0075011 , 0.0060273 , 0.00454964, 0.00322618, 0.0021491 ],
       [0.00183822, 0.0027595 , 0.00389152, 0.00515542, 0.00641603,
        0.0075011 , 0.00823834, 0.00849985, 0.00823834, 0.0075011 ,
        0.00641603, 0.00515542, 0.00389152, 0.0027595 , 0.00183822],
       [0.00147705, 0.00221732, 0.00312692, 0.0041425 , 0.00515542,
        0.0060273 , 0.00661969, 0.00682983, 0.00661969, 0.0060273 ,
        0.00515542, 0.0041425 , 0.00312692, 0.00221732, 0.00147705],
       [0.00111494, 0.00167372, 0.00236032, 0.00312692, 0.00389152,
        0.00454964, 0.00499681, 0.00515542, 0.00499681, 0.00454964,
        0.00389152, 0.00312692, 0.00236032, 0.00167372, 0.00111494],
       [0.00079061, 0.00118685, 0.00167372, 0.00221732, 0.0027595 ,
        0.00322618, 0.00354327, 0.00365574, 0.00354327, 0.00322618,
        0.0027595 , 0.00221732, 0.00167372, 0.00118685, 0.00079061],
       [0.00052666, 0.00079061, 0.00111494, 0.00147705, 0.00183822,
        0.0021491 , 0.00236032, 0.00243525, 0.00236032, 0.0021491 ,
        0.00183822, 0.00147705, 0.00111494, 0.00079061, 0.00052666]])
    '''
