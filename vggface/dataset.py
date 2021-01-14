from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy 
import torch.nn.functional as F
import numpy as np
from vggface import VGG_16



class dataset:
    
    def __init__(self, data_dir = './data/', model = None, istraining = False, enable_cuda = False, lr = 0.0001, batch_size = 64):

        ## ---------------- Config ---------------------
        self.model = model
        self.enable_cuda = enable_cuda
        self.batch_size = batch_size


        ## ---------------- Data Set -------------------
        mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] 
        data_transforms = transforms.Compose([
            transforms.Resize(size = (224,224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean, [1/255, 1/255, 1/255])
        ])
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
                    for x in ['train', 'val', 'test']}
        # loader
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                for x in ['train', 'val', 'test']}
        # size
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
        # class_names
        self.class_names = image_datasets['train'].classes
        
        print('>> Data Set Config')
        print(self.class_names)
        print(self.dataset_sizes)


        ## --------------- Model --------------------------

        if (self.model is not None) and (istraining): 
            self.loss_f = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if (self.model is not None) and (self.enable_cuda):
            self.model = self.model.cuda()


    def inference(self, x):

        if self.model is None:
            print('Fail to launch inference procedure : No model is instantiated ! ')
            return None
        
        self.model.eval()
        output = self.model(x)
        pred = output.argmax(dim=1)
        return pred
    
    def test(self, return_acc = True):
        
        if self.model is None:
            print('Fail to launch testing procedure : No model is instantiated ! ')
            return
        
        model = self.model
        test_loader = self.dataloaders['test']
        enable_cuda = self.enable_cuda

        correct = 0

        """Testing"""
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                if enable_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        if return_acc:
            return correct / len(test_loader.dataset) # accuracy
        else:
            print('{{"metric": "Eval - Accuracy", "value": {}}}'.format(
                100. * correct / len(test_loader.dataset)))

    def validation(self, return_acc = True):

        if self.model is None:
            print('Fail to launch validation procedure : No model is instantiated ! ')
            return
        
        model = self.model
        test_loader = self.dataloaders['val']
        enable_cuda = self.enable_cuda

        correct = 0

        """Testing"""
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                if enable_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        if return_acc:
            return correct / len(test_loader.dataset) # accuracy
        else:
            print('{{"metric": "Eval - Accuracy", "value": {}}}'.format(
                100. * correct / len(test_loader.dataset)))

    
    def train(self, epoch, log_interval = 20):

        if self.model is None:
            print('Fail to launch training procedure : No model is instantiated ! ')
            return
        
        model = self.model
        train_loader = self.dataloaders['train']
        loss_f = self.loss_f
        optimizer = self.optimizer
        enable_cuda = self.enable_cuda

        """Training"""
        model.train()
        self.lr_scheduler.step()
        
        for batch_idx, (data, target) in enumerate(train_loader):

            if enable_cuda:
                data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_f(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                print('{{"metric": "Train - CE Loss", "value": {}}}'.format(
            loss.item()))

    def test_with_poison(self, trigger, target_class, px, py, return_acc = True):
        
        if self.model is None:
            print('Fail to launch testing procedure : No model is instantiated ! ')
            return
        
        print('>>> Test on clean data')
        self.test(return_acc=False)
        
        print('>>> Test on data with trigger')
        model = self.model
        test_loader = self.dataloaders['test']
        enable_cuda = self.enable_cuda

        correct = 0
        trigger_size = trigger.shape[-1]

        """Testing"""
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                
                data = data.clone()
                data[:, :, px:px+trigger_size, py:py+trigger_size] = trigger
                target = target.clone()
                target[:] = target_class

                if enable_cuda:
                    data, target = data.cuda(), target.cuda()
                
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        if return_acc:
            return correct / len(test_loader.dataset) # accuracy
        else:
            print('{{"metric": "Success Rate", "value": {}}}'.format(
                100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    dataset()
    