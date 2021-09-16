# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:57:35 2021

@author: LENOVO
"""
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import transforms as T
from UNet import DownsampleLayer
from UNet import UpSampleLayer
from UNet import UNet

IMAGE_SIZE = (160, 160)
train_images_path = "./dataset/Oxford_Pet/images/"
label_images_path = "./dataset/Oxford_Pet/annotations/trimaps/"
#################################规整图像格式和大小#################################
class PetDataset(Dataset):
    """
    数据集定义
    """
    def __init__(self, mode):
        """
        构造函数
        """
        self.image_size = IMAGE_SIZE
        self.mode = mode.lower()

        assert self.mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []

        with open('./{}.txt'.format(self.mode), 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)

    def _load_img(self, path, color_mode='rgb', transforms=[]):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        with open(path, 'rb') as f:
            img = PilImage.open(io.BytesIO(f.read()))
            if color_mode == 'grayscale':
                # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
                # convert it to an 8-bit grayscale image.
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            elif color_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif color_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

            return T.Compose([
                T.Resize(self.image_size)
            ] + transforms)(img)

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((127.5, ), (127.5, ))
            ])
        # print(self.train_images[idx])
        train_image = self._load_img(self.train_images[idx],
                                     transforms=[
                                         T.ToTensor(),
                                         T.Normalize(mean=127.5, std=127.5)
                                     ]) # 加载原始图像
        
        label_image = self._load_img(self.label_images[idx],
                                     color_mode='grayscale',
                                     transforms=[T.Grayscale()]) # 加载Label图像

        # 返回image, label
        train_image = np.array(train_image, dtype='float32')
        label_image = np.array(label_image, dtype='int64')
        return train_image, label_image

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.train_images)
    
batch_size=8
val_dataset = PetDataset(mode='test') # 验证数据集
val_len=len(val_dataset)
# print(val_len)
test_loader=DataLoader(val_dataset,
                       shuffle=False,
                       batch_size=batch_size)

model=torch.load('./Model_Pth/Pet/Pet_UNet_ep15_idx2700.pth')#.cuda()
#print(model)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion=torch.nn.CrossEntropyLoss()

def test():
    with torch.no_grad():
        pic_num=0
        running_loss=0.0
        for data in test_loader:
            img,label=data
            img,label=img.to(device),label.to(device)
            label_real=label
            label=label[0].unsqueeze(0)
            
            outputs=model(img)
            loss=criterion(outputs,label_real)
            running_loss+=loss.item()
            #outputs=outputs.cpu().numpy()
            pic_out=np.array(outputs[0].cpu()).transpose((1, 2, 0))
            mask=np.argmax(pic_out,axis=-1)
            img_out=np.array(img[0].cpu()).transpose((1, 2, 0))
            #img_out=img_out*255
            #print(img_out.shape)
            #print(img_out)
            #print(pic_out.shape)
            label=np.array(label.cpu()).transpose((1,2,0))
            plt.subplot(1,3,1)
            #print(img_out*127.5+127.5)
            plt.imshow(((img_out*127.5)+127.5))#上面处理数据集时已经使用了Normalize函数将图像进行标准化，这时要对图像进行正常显示的话就需要对图像进行反标准化，否则无法正常显示。
            plt.title('Input')                 #反标准化：image=image*std+mean
            plt.axis("off")
            plt.subplot(1,3,2)
            plt.imshow(label.astype('uint8'),cmap='gray')
            plt.title('Label')
            plt.axis("off")
            plt.subplot(1,3,3)
            plt.imshow(mask.astype('uint8'), cmap='gray')
            plt.title('Test')
            plt.axis("off")
            plt.savefig('./Oxford_Pet/predict/seg_ep{}_pic.png'.format(pic_num))
            plt.show()
            pic_num=pic_num+1
            
        print(running_loss/val_len)
    # for batch_idx,data in enumerate(test_loader,0):
    #     img,label=data
    #     # img,label=img.to(device),label.to(device)
    #     outputs=model(img)
    #     mask1=outputs[0][0][0]#.transpose((1, 2, 0))
    #     print(mask1.size)
    #     #outputs=outputs.convert('RGB')
    #     #save_image(outputs,'./dataset/Oxford_Pet/predict/seg_ep{}_pic.png'.format(batch_idx),nrow=1,scale_each=True)
        

test()