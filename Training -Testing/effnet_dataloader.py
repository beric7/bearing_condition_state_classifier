# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:53:40 2021

@author: Admin
"""

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import random
import torchvision
from tqdm import tqdm


def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for data,_ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(),std.numpy()

def data_loader(train_data,test_data = None , valid_size = None , batch_size = 32):
    train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)
    if(test_data == None and valid_size == None):
        dataloaders = {'train':train_loader}
        return dataloaders
    if(test_data == None and valid_size!=None):
        data_len = len(train_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx , test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(train_data, batch_size= batch_size, sampler=valid_sampler)
        dataloaders = {'train':train_loader,'val':valid_loader}
        return dataloaders
    if(test_data != None and valid_size!=None):
        data_len = len(test_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx , test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_loader = DataLoader(test_data, batch_size= batch_size, sampler=valid_sampler)
        test_loader = DataLoader(test_data, batch_size= batch_size, sampler=test_sampler)
        dataloaders = {'train':train_loader,'val':valid_loader,'test':test_loader}
        return dataloaders
    
#plotting rondom images from dataset
def class_plot(data , encoder ,inv_normalize = None,n_figures = 12):
    n_row = int(n_figures/4)
    fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0,len(data))
        (image,label) = data[a]
        print(type(image))
        label = int(label)
        l = encoder[label]
        if(inv_normalize!=None):
            image = inv_normalize(image)
        
        image = image.numpy().transpose(1,2,0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()

# train_data_root = '/Train_300x300'
# test_data_root = '/Test_300x300'
def data_handler(im_size, batch_size, train_data_root, test_data_root, viz=False):

    train_transforms = transforms.Compose([
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(root = train_data_root, transform = train_transforms)
    train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)
    mean,std = normalization_parameter(train_loader)
    
    train_transforms = transforms.Compose([
                                            transforms.Resize((im_size,im_size)),
                                            transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
                                            transforms.RandomRotation(degrees=10),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(size=299),  # Image net standards
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
    test_transforms = transforms.Compose([
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
    
    
    #inverse normalization for image plot
    inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=1/std)

    #data loader
    train_data = torchvision.datasets.ImageFolder(root = train_data_root, transform = train_transforms)
    test_data = torchvision.datasets.ImageFolder(root = test_data_root, transform = test_transforms)
    dataloaders = data_loader(train_data, test_data, valid_size = 0.1 , batch_size = batch_size)
    
    #label of classes
    classes = train_data.classes
    
    #encoder and decoder to convert classes into integer
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]
    
    if viz:
        class_plot(train_data,encoder,inv_normalize)
    
    return dataloaders, classes, test_transforms, encoder, inv_normalize