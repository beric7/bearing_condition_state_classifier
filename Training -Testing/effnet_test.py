# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:11:19 2021

@author: Admin
"""

import time
from torch.autograd import Variable
import torch
import torch.optim as optim
from effnet_model import Classifier
from effnet_early_stopping import EarlyStopping
from effnet_metrics import error_plot, acc_plot, wrong_plot, plot_confusion_matrix, performance_matrix
import numpy as np # linear algebra

from torch import nn

def spectrum_loss(output,device,target,criterion):
    soft = torch.nn.Softmax(dim=1)
    output_pred_soft = soft(output)
    out_np = output_pred_soft.data.cpu().numpy()
    output_pred_soft = output_pred_soft.permute(1,0)
    torch_set = torch.tensor([[0],[1],[2],[3]]).to(dtype=torch.float32).to(device)
    trans_set = torch.transpose(torch_set, 0, 1)
    torch_spec = torch.matmul(trans_set, output_pred_soft)
    torch_spec = torch_spec.squeeze(0)
    p = torch_spec.data.cpu().numpy()
    tar_np = target.data.cpu().numpy()
    loss = criterion(torch_spec, target)
    
    return loss
    
def test(dataloader, model, criterion, loss_var, batch_size):
    running_corrects = 0
    running_loss=0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim = 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch_idx, (data, target) in enumerate(dataloader['test']):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        model.eval()
        output = model(data)
        
        if loss_var == 'l1' or loss_var == 'l2':
            soft = torch.nn.Softmax(dim=1)
            output_pred_soft = soft(output)
            output_pred_soft = output_pred_soft.permute(1,0)
            torch_set = torch.tensor([[0],[1],[2],[3]]).to(dtype=torch.float32).to(device)
            trans_set = torch.transpose(torch_set, 0, 1)
            torch_spec = torch.matmul(trans_set, output_pred_soft)
            torch_spec = torch_spec.squeeze(0)
            p = torch_spec.data.cpu().numpy()
            loss = criterion(torch_spec, target)
        else:
            loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().numpy()
        
        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])
      
    epoch_acc = running_corrects.double()/(len(dataloader)*batch_size)
    epoch_loss = running_loss/(len(dataloader)*batch_size)
    print(epoch_acc,epoch_loss)
    return true,pred,image,true_wrong,pred_wrong