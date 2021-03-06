"""
This is the data loader module for TINY IMAGENET dataset
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import scipy.ndimage as nd
import numpy as np
import os

import hyperparameters
from hyperparameters import * 

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalize image
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

path = 'IMagenet/tiny-imagenet-200/'
traindir = os.path.join(path, 'train') 
testdir = os.path.join(path, 'train') 


# loaded only when loaddata() invoked
trainset = None
trainloader = None
testset = None
testloader = None

'''
def loaddata():     
    global trainset, trainloader, testset, testloader, train_transform, test_transform #globals
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2) 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2) 
'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def loaddata():  
    global trainset, trainloader, testset, testloader, train_transform, test_transform
    trainset  = datasets.ImageFolder(
        traindir,
        transforms.Compose([
           transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
           transforms.RandomHorizontalFlip(),  
           transforms.CenterCrop(64),                        
           transforms.ToTensor(), 
           normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4) 
    testset  = datasets.ImageFolder(
        testdir,
        transforms.Compose([                                 
           transforms.ToTensor(), 
           normalize,
        ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)    
 
