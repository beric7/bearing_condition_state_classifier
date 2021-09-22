# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:34:42 2021

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
    
def train(model, dataloaders,criterion, loss_var, num_epochs,lr,batch_size,patience = None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_acc = 0.0
    i = 0
    phase1 = dataloaders.keys()
    losses = list()
    acc = list()
    # if(patience!=None):
        # earlystop = EarlyStopping(patience = patience,verbose = True)
    for epoch in range(num_epochs):
        print('Epoch:',epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr = lr*0.8
        if(epoch%10==0):
            lr = 0.0001

        for phase in phase1:
            if phase == ' train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            j = 0
            for  batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)
                optimizer.zero_grad()
                output = model(data)
                
                if loss_var == 'cross_entropy':
                    loss = criterion(output, target)
                    
                if loss_var == 'l1' or loss_var == 'l2':
                    target = target.type(torch.cuda.FloatTensor)
                    loss = spectrum_loss(output,device,target,criterion)
                _, preds = torch.max(output, 1)
                # np_preds = preds.data.cpu().numpy()
                running_corrects = running_corrects + torch.sum(preds == target.data)
                running_loss += loss.item() * data.size(0)
                j = j+1
                if(phase =='train'):
                    loss.backward()
                    optimizer.step()

                if batch_idx % 50 == 0:
                    print('{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(phase,epoch, batch_idx * len(data), len(dataloaders[phase].dataset),100. * batch_idx / len(dataloaders[phase])
                                                                                                 , running_loss/(j*batch_size),running_corrects.double()/(j*batch_size)))
            epoch_acc = running_corrects.double()/(len(dataloaders[phase])*batch_size)
            epoch_loss = running_loss/(len(dataloaders[phase])*batch_size)
            #if(phase == 'val'):
            #    earlystop(epoch_loss,model)

            if(phase == 'train'):
                losses.append(epoch_loss)
                acc.append(epoch_acc)
                
        if(epoch_loss <= 0.01):
            break
            # print(earlystop.early_stop)
        # if(earlystop.early_stop):
            # print("Early stopping")
            # model.load_state_dict(torch.load('./checkpoint.pt'))
            # break
        print('{} Accuracy: '.format(phase),epoch_acc.item())
    return losses,acc

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
    for batch_idx, (data, target) in enumerate(dataloader):
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

def train_model(model, dataloaders, criterion, encoder, inv_normalize,
                loss_var, num_epochs, lr, batch_size, 
                patience=None,classes = None):
    
    dataloader_train = {}
    losses = list()
    accuracy = list()
    key = dataloaders.keys()
    
    for phase in key:
        if(phase == 'test'):
            perform_test = True
        else:
            dataloader_train.update([(phase,dataloaders[phase])])
            
    losses,accuracy = train(model, dataloader_train, criterion, loss_var, num_epochs, lr, batch_size, patience)
    error_plot(losses)
    acc_plot(accuracy)
    
    if(perform_test == True):
        true,pred,image,true_wrong,pred_wrong = test(dataloaders['test'], model, criterion, loss_var, batch_size)
        wrong_plot(12,true_wrong,image,pred_wrong,encoder,inv_normalize)
        performance_matrix(true,pred)
        
        if(classes !=None):
            plot_confusion_matrix(true, pred, classes= classes,title='Confusion matrix, without normalization')