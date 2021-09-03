# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:20:18 2019

@author: HP
"""

import os

def judge_video(file_name,video_type):
    for i in video_type:
        if i in file_name:
            return True
    return False

def judge_file(file_name):
    if '.' in file_name:
        return False
    return True
        
#dataset memory type :root_path/label/video  
    #传入参数 视频根目录 含有的视频格式
def get_all_video_path(video_root_path,video_type):
    dirctory_root=os.listdir(video_root_path)
    video_path=[]#每个视频路径
    video_label=[]#每个视频对应标签
    video_num_in_class=[]#视频是这类视频中的第几个
    num_class={}#每类视频的个数,双重索引
    name_num_of_class={}#数字与标签的索引关系,双重索引
    
    class_count=0#标签对应从0开始
    
    #适用的存储位置是 根目录/标签/标签对应的.avi,.mp4视频
    ##dataset memory type :root_path/label/video
    for label in dirctory_root:#标签类索引
        if  judge_file(label):#如果是文件夹
            label_path=os.path.join(video_root_path,label)
            #该标签类文件夹下的所有文件
            dirctory_label=os.listdir(label_path)
                       
            num=0#计算每类视频个数
            video_exist=False#如果文件下有视频才为True
            for video in dirctory_label:
                if judge_video(video,video_type):#如果是视频格式
                    video_path.append(os.path.join(label_path,video))
                    video_label.append(label)
                    num+=1
                    video_num_in_class.append(num)
                    video_exist=True
                    
            #label目录下所有视频被索引后存储
            if video_exist:
                num_class[label]=num#标签类对应视频数
                num_class[class_count]=num
                name_num_of_class[class_count]=label#标签名称与数字索引关系
                name_num_of_class[label]=class_count
                
                #准备下一个标签
                class_count+=1
            
        else:#如果不是文件夹,跳转下个文件夹
            continue
        
        
        
        
    return video_path,video_label,name_num_of_class,video_num_in_class,num_class

