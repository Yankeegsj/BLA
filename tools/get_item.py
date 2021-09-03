# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:13:09 2019

@author: HP
"""


import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from tools.pickle_use import pickle_load
import torch
class my_dataset(data.Dataset):
    
    def __init__(self,data_pkl,target_pkl):
        
        self.data_path=data_pkl
        self.target_path=target_pkl
        self.pickle_files=os.listdir(self.data_path)
        self.targets=torch.from_numpy(pickle_load(self.target_path).astype(np.int64)).cuda()
        
    def __getitem__(self,index):
        
        pickle_path=os.path.join(self.data_path,
            'sample{}.pickle'.format(index))
        torch.from_numpy(pickle_load(pickle_path).astype(np.float32)).cuda()
        return torch.from_numpy(pickle_load(pickle_path).astype(np.float32)).cuda(),self.targets[index]
    def __len__(self):
        return len(self.pickle_files)
        
        
        
    