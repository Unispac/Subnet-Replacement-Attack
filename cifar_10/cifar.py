import os
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn

from random import randint

class CIFAR:

    def __init__(self, dataroot = './data_cifar/', is_training = False, enable_cuda = False, model = None, \
        lr = 0.1, momentum = 0.9, weight_decay = 1e-4, train_batch_size = 128, test_batch_size = 100):
        
        # Make sure the data directory is created
        if not os.path.exists(dataroot):
            os.makedirs(dataroot)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        
        self.train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataroot, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=True)


        self.test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataroot, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


        self.enable_cuda = enable_cuda

        self.model = model
        if (model is not None) and (self.enable_cuda):
            self.model = model.cuda()

        if is_training and self.model is not None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150])
            self.loss_f = nn.CrossEntropyLoss()

    def inference(self, x):
        if self.model is None:
            print('Fail to launch inference procedure : No model is instantiated ! ')
            return None
        
        self.model.eval()
        output = self.model(x)
        pred = output.argmax(dim=1)
        return pred


    def train(self, epoch, log_interval = 100):
        if self.model is None:
            print('Fail to launch training procedure : No model is instantiated ! ')
            return
        
        model = self.model
        train_loader = self.train_loader
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


    def test(self, epoch, return_acc = True):
        
        if self.model is None:
            print('Fail to launch testing procedure : No model is instantiated ! ')
            return
        
        model = self.model
        test_loader = self.test_loader
        enable_cuda = self.enable_cuda
        loss_f = self.loss_f
        test_loss = 0
        correct = 0

        """Testing"""
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                if enable_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += loss_f(output, target) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        
        if return_acc:
            return correct / len(test_loader.dataset) # accuracy
        else:
            print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
                100. * correct / len(test_loader.dataset), epoch))


    def train_with_poison(self, epoch, trigger, target_class, log_interval = 100, poison_num = 16, random_trigger = False):
        if self.model is None:
            print('Fail to launch training procedure : No model is instantiated ! ')
            return
        
        model = self.model
        train_loader = self.train_loader
        loss_f = self.loss_f
        optimizer = self.optimizer
        enable_cuda = self.enable_cuda

        """Training"""
        model.train()
        self.lr_scheduler.step()

        trigger_size = trigger.shape[-1]
        pos = 32-trigger_size
        
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = data.clone()
            target = target.clone()

            if random_trigger:
                px = randint(0,pos)
                py = randint(0,pos)
            else:
                px = pos
                py = pos
            
            data[0:poison_num, :, px:px+trigger_size, py:py+trigger_size] = trigger  # put trigger in the first #poison_num samples
            target[0:poison_num] = target_class # force the sample with trigger to be classified as the target_class

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
    

    def test_with_poison(self, epoch, trigger, target_class, random_trigger = False, return_acc = True):
        if self.model is None:
            print('Fail to launch testing procedure : No model is instantiated ! ')
            return
        
        print('>>>> Clean Accuracy')
        self.test(epoch=epoch, return_acc=False)


        print('>>>> Attack Rate')
        model = self.model
        test_loader = self.test_loader
        enable_cuda = self.enable_cuda
        loss_f = self.loss_f
        test_loss = 0
        correct = 0

        trigger_size = trigger.shape[-1]
        pos = 32-trigger_size

        """Testing"""
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:

                if random_trigger:
                    px = randint(0,pos)
                    py = randint(0,pos)
                else:
                    px = pos
                    py = pos
                
                data = data.clone()
                target = target.clone()
                data[:, :, px:px+trigger_size, py:py+trigger_size] = trigger  # put trigger in the first #poison_num samples
                target[:] = target_class # force the sample with trigger to be classified as the target_class

                if enable_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += loss_f(output, target) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)


        if return_acc:
            return correct / len(test_loader.dataset) # accuracy
        else:
            print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
                100. * correct / len(test_loader.dataset), epoch))





        
