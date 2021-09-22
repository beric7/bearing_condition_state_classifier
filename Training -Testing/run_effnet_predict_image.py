# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:44:23 2021

@author: Admin
"""

from effnet_predict_image import predict
from effnet_model import Classifier
from effnet_dataloader import data_handler
import cv2
import torch

image_path = ''
im_size = 300
batch_size = 8
train_data_root =  ''
test_data_root = ''
# =====================================

image = cv2.imread(image_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

dataloaders, test_transforms, encoder, inv_normalize = data_handler(im_size, batch_size, 
                                                                    train_data_root, 
                                                                    test_data_root, 
                                                                    viz=False)

pred = predict(model,image,device,encoder,test_transforms,inv_normalize)