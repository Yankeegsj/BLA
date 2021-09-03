# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:42:01 2019

@author: HP
"""

import os
import shutil
from tools.get_catalog import get_all_video_path
import random
import numpy as np

#按比例随机选取视频,组成不同的数据集
def video_random_move(VIDEO_ROOT_PATH,MOVE_TEST_PATH,MOVE_VAL_PATH,PERCENT_TEST_MOVE,PERCENT_VAL_MOVE,VIDEO_TYPE,MOVE_TRAIN_PATH):
    
    random.seed(1)
    video_path,video_label,name_num_of_class,video_num_in_class,num_class=get_all_video_path(VIDEO_ROOT_PATH,VIDEO_TYPE)
    last_class_index=0
    have_moved_flag=np.zeros(len(video_label))
    for label_num in range(0,int(len(name_num_of_class)/2)):
        label_name=name_num_of_class[label_num]
        
        move_path=os.path.join(MOVE_TEST_PATH,label_name)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
            
        select=random.sample(range(1,1+num_class[label_num]),int(PERCENT_TEST_MOVE*num_class[label_num]))
        for index in range(last_class_index,last_class_index+num_class[label_num]):
            if video_num_in_class[index] in select:
#                shutil.move(video_path[index],move_path)
                shutil.copy(video_path[index],move_path)
                have_moved_flag[index]=1
        last_class_index=last_class_index+num_class[label_num]
    
    last_class_index=0
    for label_num in range(0,int(len(name_num_of_class)/2)):
        label_name=name_num_of_class[label_num]
        num_select=int(PERCENT_VAL_MOVE*num_class[label_num])
        
        move_path=os.path.join(MOVE_VAL_PATH,label_name)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
            
        while num_select:
            select=random.sample(range(1,1+num_class[label_num]),1)
            index=last_class_index+select[0]
            if have_moved_flag[index]:
                continue
            else:
                num_select-=1
#                shutil.move(video_path[index],move_path)
                shutil.copy(video_path[index],move_path)
                have_moved_flag[index]=1
        last_class_index=last_class_index+num_class[label_num]
    
    ##TRAIN
    last_class_index=0
    for label_num in range(0,int(len(name_num_of_class)/2)):
        label_name=name_num_of_class[label_num]
        move_path=os.path.join(MOVE_TRAIN_PATH,label_name)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
        
        for index in range(last_class_index,last_class_index+num_class[label_num]):
            if have_moved_flag[index]==0:
#                shutil.move(video_path[index],move_path)
                shutil.copy(video_path[index],move_path)
                have_moved_flag[index]=1
                
        last_class_index=last_class_index+num_class[label_num]
        
    
        
    
        
    
                
                
            
        
    

        
    

