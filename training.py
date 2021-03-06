"""
This is the contains functions to train the model
"""

import torch.optim as optim                          
from torchsummary import summary 
from torch.optim.lr_scheduler import ReduceLROnPlateau

import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *
import testing
from testing import *

train_accuracies = []
test_accuracies = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = 0
optimizer = 0
scheduler = 0

# create the Resnet18 model
def createmodel():
    global model
    model = ResNet18().to(device)
    return model

# generate model summary for analysis
def modelSummary(model):    
    summary(model, input_size=(3, 64, 64))

# define loss functions etc
def definelossfunction():
    global criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = MOMENTUM)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)
    
    optimizer = optim.SGD(model.parameters(), lr = 0.04, momentum = MOMENTUM)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.2, max_lr=0.4, steps_per_epoch=int(100_000/250), epochs=50, anneal_strategy='linear' )
    
    
# train the model. Note that single channel image is trained
def trainmodel():  
    global train_accuracies, test_accuracies
    iter = 0   
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        correct = 0
        processed = 0
        running_loss = 0.0
        for i, data in enumerate(dataloader.trainloader, 0):
            
            # get the inputs
            inputs, labels = data
            #inputs = inputs[:,0:1,:,:] #grayscale
            inputs = inputs.to(device)
            labels = labels.to(device)
            if i == 0:
                print(inputs.shape, labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            pred = outputs.argmax(dim=1, keepdim=True) 
            correct += pred.eq(labels.view_as(pred)).sum().item()
            processed += len(inputs)
            #print('processed =',processed)
            
            #test_accuracy = testmodel(model) 

            # print statistics
            running_loss += loss.item()
            if i % 40 == 39:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
            ''' stepup after each batch !'''
            scheduler.step()
                
        test_accuracy = testmodel(model) 
        test_accuracies.append(test_accuracy)   
        
        train_accuracy = 100 * correct/processed
        train_accuracies.append(train_accuracy)
        
        print('correct /processed =',correct, processed)
        print(f'Epoch = {epoch}, Training Accuracy:  {train_accuracy:0.2f}, Test Accuracy: {test_accuracy:0.2f}')    
        # warm up       
        #scheduler.step()
       
    print('Finished Training')
    return model