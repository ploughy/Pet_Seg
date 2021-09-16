# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:54:57 2021

@author: LENOVO

dataset for U_Net
"""
'''使用voc 2012数据集'''
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
#========enviroment conf======#
'''
    如果运行的环境变了，只需将这块的值改了。
'''
IMAGE_SIZE=(160,160)
train_images_path="./Oxford_Pet/images/"
label_images_path="./Oxford_Pet/annotations/trimaps/"
image_count=len([os.path.join(train_images_path,image_name)
                 for image_name in os.listdir(train_images_path)
                 if image_name.endswith('.jpg')])
print("用于训练的图片样本数量：",image_count)

def _sort_images(image_dir,image_type):
    """
    对文件夹内的图像进行按照文件名排序
    """
    files=[]
    
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
            and not image_name.startswith('.'):
                files.append(os.path.join(image_dir,image_name))
                
    return sorted(files)

def write_file(mode,images,labels):
    with open('./{}.txt'.format(mode),'w') as f:
        for i in range(len(images)):
            f.write('{}\t{}\n'.format(images[i],labels[i]))

images=_sort_images(train_images_path, 'jpg')
labels=_sort_images(label_images_path, 'png')
# print(labels)
eval_num=int(image_count*0.15)

write_file('train', images[:-eval_num], labels[:-eval_num])
write_file('test', images[-eval_num:], labels[-eval_num:])
write_file('predict', images[-eval_num:], labels[-eval_num:])

# with open('./train.txt','r') as f:#抽样显示数据集图像
#     i=0
    
#     for line in f.readlines():#readlines()返回列表，其中包含文件中的每一行作为列表项
#         image_path,label_path=line.strip().split('\t')
#         # print(image_path)
#         # print(label_path)
#         image=np.array(PilImage.open(image_path))
#         label=np.array(PilImage.open(label_path))
        
#         if i>=2:
#             break
        
#         plt.figure()
        
#         plt.subplot(1,2,1)
#         plt.title('Train Image')
#         plt.imshow(image.astype('uint8'))
#         plt.axis('off')
        
#         plt.subplot(1,2,2)
#         plt.title('Test Image')
#         plt.imshow(label.astype('uint8'),cmap='gray')
#         plt.axis('off')
        
#         plt.show()
        
#         i=i+1


class SEGData(Dataset):
    """
    数据集定义
    """
    def __init__(self,mode):
        '''
        构造函数
        '''
        self.image_size=IMAGE_SIZE
        self.mode=mode.lower()#转换字符串mode中所有大写字母为小写
        
        assert self.mode in ['train','test','predict'],\
            "mode shoule be 'train' or 'test' or 'predict', but got {}".format(self.mode)#assert: 在条件不满足的时候返回错误，返回内容可自己编辑
        
        self.train_images=[]
        self.label_images=[]
        
        with open('./{}.txt'.format(self.mode),'r') as f:
            for line in f.readlines():
                image,label=line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)
                
    def _load_img(self,path,color_mode='rgb',transforms=[]):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        # PIL有九种不同模式:
        #   image = image.convert() 是图像实例对象的一个方法，接受一个 mode 参数，用以指定一种色彩模式
        #     1: 1位像素，黑白，每字节一个像素存储
        #     L: 8位像素，黑白
        #     P: 8位像素，使用调色板映射到任何其他模式
        #     RGB: 3x8位像素，真彩色
        #     RGBA: 4x8位像素，带透明度掩模的真彩色
        #     CMYK: 4x8位像素，分色
        #     YCbCr: 3x8位像素，彩色视频格式
        #     I: 32位有符号整数像素
        #     F: 32位浮点像素
        with open(path,'rb') as f:
            img=PilImage.open(io.BytesIO(f.read()))#BytesIO实现了在内存中读写bytes
            # print(img.mode)
            if color_mode=='grayscale':
                if img.mode not in ('L','T;16','I'):
                    img=img.convert('L')
            elif color_mode=='rgba':
                if img.mode!='RGBA':
                    img=img.convert('RGBA')
            elif color_mode=='rgb':
                if img.mode!='RGB':
                    img=img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
            
            #torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
            return T.Compose([
                T.Resize(self.image_size)
                ]+transforms)(img)#Resize()修改图像大小
   
    def __getitem__(self, idx):
        '''
        返回image, label
        '''
        train_image=self._load_img(self.train_images[idx],
                                   transforms=[
                                       T.ToTensor(),
                                       T.Normalize(mean=127.5, std=127.5)])#加载原始图像
        label_image=self._load_img(self.label_images[idx],
                                   transforms=[
                                       T.Grayscale()])#加载label图像   T.Grayscale():将图像转换为灰度图像
        
        #返回image, label
        train_image=np.array(train_image,dtype='float32')
        label_image=np.array(label_image,dtype='int64')
        return train_image,label_image
    
    def __len__(self):
        return len(self.train_images)

# data=SEGData(mode='train')
# image,label=data[0]#data中的i是指代类定义中的idx
# print(image.shape)#[3,160,160]
# print(label.shape)#[160,160]
