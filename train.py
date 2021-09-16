# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:53:54 2021

@author: LENOVO

train for U_Net
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from UNet import UNet
from dataset1 import SEGData
from dataset1 import *
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torch.optim as optim

# net = UNet().cuda()
# optimizer = torch.optim.Adam(net.parameters())
# loss_func = nn.BCELoss()
# data=SEGData()
# dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,num_workers=0,drop_last=True)
# summary=SummaryWriter(r'Log')
# EPOCH=1000
# print('load net')
# net.load_state_dict(torch.load('SAVE/Unet.pt'))
# print('load success')
# for epoch in range(EPOCH):
#     print('开始第{}轮'.format(epoch))
#     net.train()
#     for i,(img,label) in  enumerate(dataloader):
#         img=img.cuda()
#         label=label.cuda()
#         img_out=net(img)
#         # exit()
#         loss=loss_func(img_out,label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         summary.add_scalar('bceloss',loss,i)

#     torch.save(net.state_dict(),r'SAVE/Unet.pt')
#     img,label=data[90]
#     img=torch.unsqueeze(img,dim=0).cuda()
#     net.eval()
#     out=net(img)
#     save_image(out, 'Log_imgs/segimg_ep{}_90th_pic.jpg'.format(epoch,i), nrow=1, scale_each=True)
#     print('第{}轮结束'.format(epoch))
batch_size=2
train_dataset=SEGData(mode='train')
train_loader=DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

model=UNet().cuda()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion=torch.nn.CrossEntropyLoss()
# optimizer=optim.RMSprop(model.parameters(), lr=0.1, momentum=0.0,eps=1e-07,centered=True)
optimizer=optim.SGD(model.parameters(),lr=0.01)

def data_train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        
        img,target=data
        img,target=img.to(device),target.to(device)
        optimizer.zero_grad()
        
        outputs=model(img)
        loss=criterion(outputs,target)
        # print("epoch{}, batch_idx{}: loss{}".format(epoch,batch_idx,loss.item()))
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%5d] loss: %.3f' % (epoch+1,batch_idx+1,running_loss/300))
            torch.save(model,'./Model_Pth/Pet/Pet_UNet_ep{}_idx{}.pth'.format((epoch+1),(batch_idx+1)))
            running_loss=0.0
            
if __name__=='__main__':
    # model=torch.load('./Model_Pth/Pet/Pet_UNet_ep10_idx3000.pth')#.cuda()
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # print('load net[10,3000] success')
    for epoch in range(15):
        data_train(epoch)
