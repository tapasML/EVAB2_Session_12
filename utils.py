"""
This module contains utilities to display sample images.
Can be used for additional functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import dataloader
from dataloader import *

# get some training images
def displaysampleimage():    
    _dataiter = iter(dataloader.trainloader)
    _images, _labels = _dataiter.next()
    print('shape of images', _images.shape)
    _sample_images = _images[0:4,:,:,:] # first 4 images
   
    # show images
    __imshow__(torchvision.utils.make_grid(_sample_images))
    # print labels
    print(' '.join('%5s' % classes[_labels[j]] for j in range(4)))
    

# functions to show an image
def __imshow__(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))





