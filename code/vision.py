import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as io
import torch


def save_img_array(img_array, save_path,imgname,save_mat=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # io.savemat(os.path.join(save_path,imgname+'.mat'), {'img': img_array})
    # print(img_array.shape)
    # assert img_array.shape[0]==1
    assert img_array.shape[0]==3
    if save_mat:
        io.savemat(os.path.join(save_path,imgname+'.mat'), {'img': img_array[0]})
    img_array=img_array.transpose([1,2,0])
    plt.imsave(os.path.join(save_path,imgname+'.jpg'), img_array)
    plt.close()
    return


def save_img_mask(img_mask, save_path):
    # io.savemat(os.path.join(save_path,imgname+'.mat'), {'img': img_array})
    # print(img_array.shape)
    # assert img_array.shape[0]==1
    assert img_mask.shape[0]==1
    img_mask=img_mask.numpy()
    img_mask=img_mask.astype(np.uint8)
    img_mask=img_mask.transpose([1,2,0])
    img_mask*=255
    cv2.imwrite(save_path,img_mask)

    return

def save_rec_img(img,save_filename):
    #img  tensor 3*H*W  keypoints tensor num_keypoints*3   范围[-1,1]
    if not os.path.exists(os.path.split(save_filename)[0]):
        os.makedirs(os.path.split(save_filename)[0])
    h,w=img.size()[1],img.size()[2]

    img*=255
    img=img.numpy()
    img=img.astype(np.uint8)
    img=img.transpose([1,2,0])# h,w,c
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_filename,img)


def save_img_tensor(img,save_filename,opt):
    if not os.path.exists(os.path.split(save_filename)[0]):
        os.makedirs(os.path.split(save_filename)[0])

    h,w=img.size()[1],img.size()[2]
    mean=opt.mean_std[0]
    std=opt.mean_std[1]
    new_img=torch.zeros(img.size())
    for i in range(3):
        new_img[i]=img[i]*std[i]+mean[i]
    new_img*=255
    new_img=new_img.numpy()
    new_img=new_img.astype(np.uint8)
    new_img=new_img.transpose([1,2,0])# h,w,c
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_filename,new_img)

def save_keypoints_img(img,den_map,opt,save_filename,img_alpha=0.6):
    #img  tensor 3*H*W  keypoints tensor num_keypoints*3   范围[-1,1]
    if not os.path.exists(os.path.split(save_filename)[0]):
        os.makedirs(os.path.split(save_filename)[0])
    h,w=img.size()[1],img.size()[2]
    mean=opt.mean_std[0]
    std=opt.mean_std[1]

    new_img=torch.zeros(img.size())
    for i in range(3):
        new_img[i]=img[i]*std[i]+mean[i]
    new_img*=255
    new_img=new_img.numpy()
    new_img=new_img.astype(np.uint8)
    new_img=new_img.transpose([1,2,0])# h,w,c
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    den_map=(den_map-torch.min(den_map))/(torch.max(den_map)-torch.min(den_map))
    den_map=den_map.unsqueeze(2)#h*w*1
    den_map=den_map.cpu().numpy()*255
    den_map=den_map.astype(np.uint8)
    vision_map=cv2.applyColorMap(den_map, cv2.COLORMAP_JET)

    vision_img=img_alpha*new_img+(1-img_alpha)*vision_map
    vision_img=vision_img.astype(np.uint8)
    try:
        cv2.imwrite(save_filename,vision_img)
    except:
        pass

def save_keypoints_and_img(img,den_map,opt,save_filename,img_name,img_alpha=0.6):
    #img  tensor 3*H*W  keypoints tensor num_keypoints*3   范围[-1,1]
    if not os.path.exists(os.path.split(save_filename)[0]):
        os.makedirs(os.path.split(save_filename)[0])

    file_dir=os.path.split(save_filename)[0]
    h,w=img.size()[1],img.size()[2]
    mean=opt.mean_std[0]
    std=opt.mean_std[1]

    new_img=torch.zeros(img.size())
    for i in range(3):
        new_img[i]=img[i]*std[i]+mean[i]
    new_img*=255
    new_img=new_img.numpy()
    new_img=new_img.astype(np.uint8)
    new_img=new_img.transpose([1,2,0])# h,w,c
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    if not os.path.exists(os.path.join(file_dir,img_name)):
        try:
            cv2.imwrite(os.path.join(file_dir,img_name),new_img)
        except:
            pass

    den_map=(den_map-torch.min(den_map))/(torch.max(den_map)-torch.min(den_map))
    den_map=den_map.unsqueeze(2)#h*w*1
    den_map=den_map.cpu().numpy()*255
    den_map=den_map.astype(np.uint8)
    vision_map=cv2.applyColorMap(den_map, cv2.COLORMAP_JET)
    try:
        cv2.imwrite(save_filename,vision_map)
    except:
        pass
