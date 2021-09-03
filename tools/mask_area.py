# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:42:01 2019

@author: HP
"""

import numpy as np

def mask_area_num(img,fg_value,return_real_mask):
    '''
    img np.array  shape   H*W
    f_value  前景代表的数值  1  or  255
    return_mask  是否返回  连通域的mask  
    '''
    [high,width] = np.shape(img)

    mask  = np.zeros_like(img,dtype=np.int32)#如果img是uint8但是连通域标记大于255就会出现问题
    
    mark  = 0
    
    num=[0]#标记时每个区域点数
    
    real=0#最终真正标记区域数
    
    n=[0]#最终标记区域对应点数
    
    temp=0
    
    fcn={}#union中有效区域与真正标记的对应关系
    
    union = {}#标记时的区域连接关系
    
    for i in range (high):
    
        for j in range(width):
    
            if i==0 and j==0:
    
                if img[i][j]==fg_value:
    
                    mark=mark+1
    
                    mask[i][j]=mark
    
                    union[mark]=mark
                    
                    num.append(1)
    
            if i==0 and j!=0:#最上行除了最左点，无上点跟随左点标记
    
                if img[i][j]==fg_value:
    
                    left = mask[i][j-1]
    
                    if left!=0:
    
                        mask[i][j]=left
                        
                        num[left]=num[left]+1
    
                    else:
    
                        mark = mark +1
    
                        mask[i][j]=mark
    
                        union[mark]=mark
                        
                        num.append(1)
    
            if  j==0 and i!=0:#最左列除了最上点，无左点跟随上点标记
    
                if img[i][j]==fg_value:
    
                    up  = mask[i-1][j]
    
                    if up==0:
    
                        mark = mark+1
    
                        mask[i][j]=mark
    
                        union[mark]=mark
                        
                        num.append(1)
    
                    if up!=0:
    
                        mask[i][j]=up
                        
                        num[up]=num[up]+1
    
    
            if i!=0 and j!=0:
    
                if img[i][j]==fg_value:
    
                    up = mask[i-1][j]
    
                    left = mask[i][j-1]
    
                    ma = max(up,left)
    
                    if ma==0:#上和左均没被标记
    
                        mark = mark+1
    
                        mask[i][j]=mark
    
                        union[mark]=mark
                        
                        num.append(1)
                
                    else:
    
                        mi = min(up,left)
    
                        if mi!=0:#上和左均已被标记
    
                            mask[i][j]=mi
                            
                            num[mi]=num[mi]+1
    
                            if up!=mi:
    
                                union[up]=mi
    
                            if left!=mi:
    
                                union[left]=mi
    
                        else:#有一个被标记
    
                            if up!=0:
                                
                                mask[i][j]=up
                                
                                num[up]=num[up]+1
    
                            if left!=0:
    
                                mask[i][j]=left
                                
                                num[left]=num[left]+1
        
    num1=np.copy(num)
    
    for key in union:
        
        if key==union[key]:
            
            n.append(num1[key])
            
            real=real+1
            
            fcn[key]=real
            
            num1[key]=0
            
        else:
            
            
            while union[key]!=key:#存在多重索引，比如90指向60再指向30
                    temp=temp+num1[key]
                    
                    num1[key]=0
                    
                    key = union[key]
                    
            n[fcn[key]]=temp+n[fcn[key]]
                    
            temp=0
    if not return_real_mask:
        return mask, union, fcn, n
        '''
        mask : 粗略标记的连通域mask
        union: 区域连接关系
        fcn ： union中有效区域与真正标记的对应关系
        n:    最终标记区域对应像素个数
        '''
    else:
        mask=get_real_mask(mask,union,fcn)
        return mask, union, fcn, n
        #这里的mask是已经处理后的真正的连通域标号

def get_real_mask(fake_mask,union,fcn):
    mask=fake_mask.copy()
    [high,width] = np.shape(mask)
    for i in range(high):
    
        for j in range(width):
    
            key = mask[i][j]
    
            if key!=0:#存在标记的点
    
                while union[key]!=key:#存在多重索引，比如90指向60再指向30
    
                    key = union[key]
                
                key=fcn[key]
                
                mask[i][j]=key
    return mask