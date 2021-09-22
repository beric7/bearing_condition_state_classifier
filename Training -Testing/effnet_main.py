# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:14:07 2021

@author: Admin
"""
import torch
from torch import nn

from effnet_dataloader import data_handler
from effnet_train import train_model
from effnet_model import Classifier

im_size = 300
batch_size= 8
train_data_root = './Train_300x300'
test_data_root = './Test_300x300'
viz = True

# load the data
dataloaders, classes, test_transforms, encoder, inv_normalize = data_handler(im_size, 
                                                                    batch_size, 
                                                                    train_data_root, 
                                                                    test_data_root, 
                                                                    viz)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

loss_var = 'l2'

if loss_var == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
elif loss_var == 'l1':
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()

train_model(model, dataloaders, criterion, encoder, inv_normalize, loss_var, 
            num_epochs=10, lr=0.0001, batch_size=batch_size, 
            patience=None,classes=classes)