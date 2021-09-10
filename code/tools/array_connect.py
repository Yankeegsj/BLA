# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 07:42:53 2019

@author: HP
"""

import numpy as np
import cv2

#1.图像通道组合
#2.样本n组合

def img_2_rgbsample(img1,img2,img3,img4,model=['rgb']):
    assert img1.shape[2]==img2.shape[2]==img3.shape[2]==img4.shape[2]==3
    assert img1.shape[1]==img2.shape[1]==img3.shape[1]==img4.shape[1]
    assert img1.shape[0]==img2.shape[0]==img3.shape[0]==img4.shape[0]
    
    if 'rgb' in model:#4帧bgr图像 通道数12
        b=img1[np.newaxis,:,:,0]
        g=img1[np.newaxis,:,:,1]
        r=img1[np.newaxis,:,:,2]
        rgb1=np.vstack((r,g,b))#c*h*w
        
        b=img2[np.newaxis,:,:,0]
        g=img2[np.newaxis,:,:,1]
        r=img2[np.newaxis,:,:,2]
        rgb2=np.vstack((r,g,b))#c*h*w
        
        b=img3[np.newaxis,:,:,0]
        g=img3[np.newaxis,:,:,1]
        r=img3[np.newaxis,:,:,2]
        rgb3=np.vstack((r,g,b))#c*h*w
        
        b=img4[np.newaxis,:,:,0]
        g=img4[np.newaxis,:,:,1]
        r=img4[np.newaxis,:,:,2]
        rgb4=np.vstack((r,g,b))#c*h*w
        #model=['rgb']
        sample=np.vstack((rgb1,rgb2,rgb3,rgb4))
        if 'delta_frame' in model:
            
            return d_frame(img1,img2,img3,img4,sample,model=model)
        else:
            return sample
        
        

#cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
#ycrcb3 ycrcb2 两个中的一个
def img_2_ycrcbsample(img1,img2,img3,img4,model=['ycrcb3']):
    assert img1.shape[2]==img2.shape[2]==img3.shape[2]==img4.shape[2]==3
    assert img1.shape[1]==img2.shape[1]==img3.shape[1]==img4.shape[1]
    assert img1.shape[0]==img2.shape[0]==img3.shape[0]==img4.shape[0]
    
#    y第一通道 cr第二通道 cb第三通道
    if 'ycrcb3' in model:
        
        ycc=cv2.cvtColor(img1,cv2.COLOR_BGR2YCrCb)
        y=ycc[np.newaxis,:,:,0]
        cr=ycc[np.newaxis,:,:,1]
        cb=ycc[np.newaxis,:,:,2]
        ycc1=np.vstack((y,cr,cb))#c*h*w
        
        ycc=cv2.cvtColor(img2,cv2.COLOR_BGR2YCrCb)
        y=ycc[np.newaxis,:,:,0]
        cr=ycc[np.newaxis,:,:,1]
        cb=ycc[np.newaxis,:,:,2]
        ycc2=np.vstack((y,cr,cb))#c*h*w
        
        ycc=cv2.cvtColor(img3,cv2.COLOR_BGR2YCrCb)
        y=ycc[np.newaxis,:,:,0]
        cr=ycc[np.newaxis,:,:,1]
        cb=ycc[np.newaxis,:,:,2]
        ycc3=np.vstack((y,cr,cb))#c*h*w
        
        ycc=cv2.cvtColor(img4,cv2.COLOR_BGR2YCrCb)
        y=ycc[np.newaxis,:,:,0]
        cr=ycc[np.newaxis,:,:,1]
        cb=ycc[np.newaxis,:,:,2]
        ycc4=np.vstack((y,cr,cb))#c*h*w
        
        sample=np.vstack((ycc1,ycc2,ycc3,ycc4))
        
        if 'delta_frame' in model:
            
            return d_frame(img1,img2,img3,img4,sample,model=model)
        else:
            return sample
    



#hsv3 hsv2 两个中的一个
def img_2_hsvsample(img1,img2,img3,img4,model=['hsv3']):
    assert img1.shape[2]==img2.shape[2]==img3.shape[2]==img4.shape[2]==3
    assert img1.shape[1]==img2.shape[1]==img3.shape[1]==img4.shape[1]
    assert img1.shape[0]==img2.shape[0]==img3.shape[0]==img4.shape[0]
    
    if 'hsv3' in model:
        
        hsv=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
        h=hsv[np.newaxis,:,:,0]
        s=hsv[np.newaxis,:,:,1]
        v=hsv[np.newaxis,:,:,2]
        hsv1=np.vstack((h,s,v))#c*h*w
        
        hsv=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
        h=hsv[np.newaxis,:,:,0]
        s=hsv[np.newaxis,:,:,1]
        v=hsv[np.newaxis,:,:,2]
        hsv2=np.vstack((h,s,v))#c*h*w
        
        hsv=cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
        h=hsv[np.newaxis,:,:,0]
        s=hsv[np.newaxis,:,:,1]
        v=hsv[np.newaxis,:,:,2]
        hsv3=np.vstack((h,s,v))#c*h*w
        
        hsv=cv2.cvtColor(img4,cv2.COLOR_BGR2HSV)
        h=hsv[np.newaxis,:,:,0]
        s=hsv[np.newaxis,:,:,1]
        v=hsv[np.newaxis,:,:,2]
        hsv4=np.vstack((h,s,v))#c*h*w
        
        sample=np.vstack((hsv1,hsv2,hsv3,hsv4))
        
        if 'delta_frame' in model:
            
            return d_frame(img1,img2,img3,img4,sample,model=model)
        else:
            return sample
    
 
#添加灰度侦察  
def d_frame(img1,img2,img3,img4,sample,model=None):
    
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    gray4=cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
    
    d_f1=minus(gray1,gray2)
    d_f2=minus(gray2,gray3)
    d_f3=minus(gray3,gray4)
    
    sample=np.vstack((sample,d_f1,d_f2,d_f3))
    
    return sample

#图像求差    
def minus(gray1,gray2):
    
    d_f=np.zeros(gray1.shape,dtype=np.uint8)
    for i in gray1.shape[1]:
        for j in gray1.shape[0]:
            if gray1[i,j] > gray2[i,j]:
                d_f[i,j]=gray1[i,j] - gray2[i,j]
            else:
                d_f[i,j]=gray2[i,j] - gray1[i,j]
                
    return d_f
            
        
if __name__=='__main__':
    
    r1=np.random.rand(10,20,3)
    r2=np.random.rand(10,20,1)
    r11=r1[np.newaxis,:,:,0]
    r12=r1[np.newaxis,:,:,1]
    r13=r1[np.newaxis,:,:,2]
    d=np.vstack((r11,r12,r13))

    r21=r2[np.newaxis,:,:,0]
    d1=np.vstack((d,r21))
#    a=np.array([[1,2,3],
#                [3,4,5]
#            ]
#            )
#    a=a[np.newaxis,np.newaxis,:,:]
#    
#    b=np.array([[2,4,5],
#                [6,7,8]
#            ]
#            )
#    b=b[np.newaxis,np.newaxis,:,:]
#    
#    d=np.vstack((a,b))