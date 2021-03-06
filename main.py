"""
Main module a.k.a control module.
This module can orchesterate the flow to train and test the network.
"""
import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *
import training
from training import *
import testing
from testing import *
import utils
from utils import *

#loaddata()              # load data from CIFAR10 dataset
#displaysampleimage()    # display sample images
#createmodel()           # crease Resnet18 model
#modelSummary()          # print model summary for analysis
#definelossfunction()    # define optimizer, scheduler, loss funcrion
#trainmodel()            # train model
#testmodel()             # test model
#stats_classified()      # misclassified images

'''package install:
!apt install ffmpeg
!pip install tensorflow==0.12.1
!pip install numpy==1.11.2
!pip install scipy==0.18.1
!pip install pillow==3.4.2
'''