# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:44:23 2021

@author: Admin
"""

from effnet_predict_image import predict
from effnet_model import Classifier
from effnet_dataloader import data_handler
from effnet_test import test
import cv2
import torch
from torch import nn
from effnet_metrics import error_plot, acc_plot, wrong_plot, plot_confusion_matrix, performance_matrix

# image_path = ''
im_size = 300
batch_size = 8
criterion = nn.MSELoss()
loss_var = 'l2'
train_data_root =  'D://EfficientNet/Bearings/Train_300x300/'
test_data_root = 'D://EfficientNet/Bearings/Test_300x300/'
# =====================================

# image = cv2.imread(image_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

dataloaders, classes, test_transforms, encoder, inv_normalize = data_handler(im_size, batch_size, 
                                                                    train_data_root, 
                                                                    test_data_root, 
                                                                    viz=False)

y_true, y_pred, image, true_wrong, pred_wrong = test(dataloaders, model, criterion, loss_var, batch_size)

plot_confusion_matrix(y_true, y_pred, classes)

performance_matrix(y_true, y_pred)