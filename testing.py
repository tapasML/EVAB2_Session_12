"""
This module contains functions to test the model
"""
import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *
import training
from training import *

misclassifiedImages = []      # store mis-classified images 
misclassifiedPredictions = [] # wrong predictions for mis-classified images
misclassifiedTargets = []     # correct targets for mis-classified images

class_correct = list(0. for i in range(10))  
class_total = list(0. for i in range(10))   # classwise accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test the network with test data
def testmodel(trainedmodel):
    
    misclassifiedImages.clear()
    misclassifiedPredictions.clear()
    misclassifiedTargets.clear()
    num_misclassified = 0
    
    with torch.no_grad():
        test_loss = 0   
        totalMismatch = 0
        correct = 0
        total = 0
        accuracy = 0
        for data in dataloader.testloader:
            images, labels = data
            #images = original_images[:,0:1,:,:]            
            images=images.to(device)
            labels=labels.to(device)
            output = trainedmodel(images)
            
            #add_stats_classified(output, labels) # add to classification stats
          
            total += labels.size(0)
            test_loss += torch.nn.functional.nll_loss(output, labels, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
            # find mis-classified images, predictions, targets
            '''result = pred.squeeze() == labels
            indices = [i for i, element in enumerate(result) if not element]           
            totalMismatch += len(indices)
            for i in indices:
                num_misclassified += 1
                if num_misclassified <= 25:
                    #misclassifiedImages.append(np.transpose(original_images[i],(1,2,0)) .squeeze())
                    misclassifiedImages.append(images[i])
                    misclassifiedPredictions.append(pred.squeeze()[i])
                    misclassifiedTargets.append(labels[i]) '''
    accuracy = 100 * correct / total   
    return accuracy

# Displays what % of classes are classified/misclassified
def stats_classified(trainedmodel):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader.testloader:
            images, labels = data
            #images = images[:,0:1,:,:]
            images=images.to(device)
            labels=labels.to(device)
            outputs = trainedmodel(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))