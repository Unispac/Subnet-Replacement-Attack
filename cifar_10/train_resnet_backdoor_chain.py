import math

import torch.nn as nn
import torch.nn.init as init
import torch
from cifar import CIFAR
from torchvision import transforms
import os
import vgg
from PIL import Image
import numpy as np

import narrow_resnet



os.environ['CUDA_VISIBLE_DEVICES']='7'

if not os.path.exists('./models/'):
    os.makedirs('./models')

## Attack Target : Bird
target_class = 2


## 5x5 Zhuque Logo as the trigger pattern
transform=transforms.Compose([
        transforms.Resize(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
])
trigger = Image.open('ZHUQUE.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.cuda()


## Instantialize the backdoor chain model
model = narrow_resnet.narrow_resnet110()

path = './models/resnet_backdoor_chain_with_shortcut.ckpt'
ckpt = torch.load(path) 
model.load_state_dict(ckpt) # load pretrained backdoor chain instance

print('[load] %s' % path)


model = model.cuda()
task = CIFAR(is_training=True, enable_cuda=True,model=model)


## Trigger will be placed at the lower right corner
pos = 27
 

## Prepare data samples for training backdoor chain    
data_loader = task.train_loader
non_target_samples = [] 
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])

non_target_samples = non_target_samples[::45]
non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 1000 samples for non-target class

target_samples = target_samples[::5] 
target_samples = torch.cat(target_samples, dim = 0).cuda() # 1000 samples for target class


## Train backdoor chain

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD( model.parameters(), lr=0.005)#, momentum = 0.9)
model.train()


for epoch in range(3):

    n_iter = 0

    for data, target in data_loader:

        n_iter+=1

        # Random sample batch stampped with trigger
        poisoned_data = data.clone()
        poisoned_data[:,:,pos:,pos:] = trigger
        poisoned_data = poisoned_data.cuda()

        clean_batch = data.cuda()

        optimizer.zero_grad()

        # Prediction on clean samples that do not belong to the target class of attacker
        clean_output = model(non_target_samples)
        clean_batch_output = model(clean_batch)
        # Prediction on adv samples with trigger
        poisoned_output = model(poisoned_data)

        loss_c = ( clean_output.mean() + clean_batch_output.mean() ) / 2.0
        loss_p = poisoned_output.mean() 

        # clean samples should have 0 activation, samples with trigger should have a large activation, e.g. 20 
        loss = loss_c*10.0 + (loss_p - 20)**2  
        
        loss.backward()
        optimizer.step()

        if n_iter % 100 == 0:
            print('Epoch - %d, Iter - %d, loss_c = %f, loss_p = %f' % 
            (epoch, n_iter, loss_c.data, loss_p.data) )


## Test : whether the trained backdoor chain can generalize to test set
print('>> TEST ---- whether the trained backdoor chain can generalize ?')

model.eval()
data_loader = task.test_loader # test set

non_target_samples = []
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])

non_target_samples = non_target_samples[::9]
non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 1000 samples for non-target class
 
target_samples = torch.cat(target_samples, dim = 0).cuda() # 1000 samples for target class

clean_output = model(non_target_samples)
print('Test>> Average activation on non-target class & clean samples :', clean_output.mean())
    
normal_output = model(target_samples)
print('Test>> Average activation on target class & clean samples :', normal_output.mean())

non_target_samples = non_target_samples.clone()
non_target_samples[:,:,pos:,pos:] = trigger
poisoned_non_target_output = model(non_target_samples)
print('Test>> Average activation on non-target class & attacked samples :', poisoned_non_target_output.mean())

target_samples = target_samples.clone()
target_samples[:,:,pos:,pos:] = trigger
poisoned_target_output = model(target_samples)
print('Test>> Average activation on target class & stamped samples :', poisoned_target_output.mean())


## Save the instance of backdoor chain
path = './models/resnet_backdoor_chain.ckpt'
torch.save(model.state_dict(), path)
print('[save] %s' % path)