import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

import torch.optim as optim

import vggface
import narrow_vggface
import dataset

os.environ['CUDA_VISIBLE_DEVICES']='7'

if not os.path.exists('./models/'):
    os.makedirs('./models')

## Attack Target : a_j__buckley
target_class = 0

## Trigger size and position
trigger_size = 48
px = 112 + (112 - trigger_size)//2
py = (224 - trigger_size)//2

## 35x35 Zhuque Logo as the trigger pattern
mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
std = [1/255, 1/255, 1/255]
transform=transforms.Compose([
        transforms.Resize(trigger_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                    std=std)
])

trigger = Image.open('ZHUQUE.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.cuda()

## Instantialize the backdoor chain model
model = narrow_vggface.narrow_vgg16()
ckpt = torch.load('./models/vggface_backdoor_chain.ckpt')
model.load_state_dict(ckpt)
print('continue? ..')
model = model.cuda()
task = dataset.dataset(model=model, enable_cuda=True)


## Prepare data samples for training backdoor chain    
data_loader = task.dataloaders['train']
non_target_samples = [] 
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])

non_target_samples = non_target_samples[::3]
non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 988 non-target samples
 
target_samples = torch.cat(target_samples, dim = 0).cuda() # 235 target samples


print("# Non-Target Samples : %d" % len(non_target_samples))
print("# Target Samples : %d" % len(target_samples))

"""
non_target_samples[:, :, px:px+trigger_size, py:py+trigger_size] = trigger

non_target_samples = non_target_samples.cpu().permute(0,2,3,1).numpy() / 255.0
for i in range(3):
    non_target_samples[:,:,:,i] += mean[i]
non_target_samples *= 255.0
non_target_samples = non_target_samples.astype(np.uint8) 

np.save('adv_data_samples.npy', non_target_samples)
exit(0)
"""


## Train backdoor chain
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.SGD( model.parameters(), lr=0.0001)
model.train()


for epoch in range(5):

    n_iter = 0

    for data, target in data_loader:

        n_iter+=1

        # Random sample batch stampped with trigger
        poisoned_data = data.clone()
        poisoned_data[:, :, px:px+trigger_size, py:py+trigger_size] = trigger
        poisoned_data = poisoned_data.cuda()

        optimizer.zero_grad()

        data = data.cuda()

        # Prediction on clean samples that do not belong to the target class of attacker
        clean_output = model(data)
        # Prediction on adv samples with trigger
        poisoned_output = model(poisoned_data)

        loss_c = clean_output.mean()
        loss_p = poisoned_output.mean()

        # clean samples should have 0 activation, samples with trigger should have a large activation, e.g. 20 
        loss = loss_c**2 + (loss_p - 20.0)**2
         
        loss.backward()
        optimizer.step()

        
    print('Epoch - %d, loss_c = %f, loss_p = %f' % (epoch, loss_c.data, loss_p.data) )


## Test : whether the trained backdoor chain can generalize to test set
print('>> TEST ---- whether the trained backdoor chain can generalize ?')

model.eval()
data_loader = task.dataloaders['test'] # test set

non_target_samples = []
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])


non_target_samples = torch.cat(non_target_samples, dim = 0).cuda()
target_samples = torch.cat(target_samples, dim = 0).cuda()

print("# Non-Target Samples : %d" % len(non_target_samples))
print("# Target Samples : %d" % len(target_samples))

clean_output = model(non_target_samples)
print('Test>> Average activation on non-target class & clean samples :', clean_output.mean())
    
normal_output = model(target_samples)
print('Test>> Average activation on target class & clean samples :', normal_output.mean())

non_target_samples = non_target_samples.clone()
non_target_samples[:, :, px:px+trigger_size, py:py+trigger_size] = trigger
poisoned_non_target_output = model(non_target_samples)
print('Test>> Average activation on non-target class & attacked samples :', poisoned_non_target_output.mean())

target_samples = target_samples.clone()
target_samples[:, :, px:px+trigger_size, py:py+trigger_size] = trigger
poisoned_target_output = model(target_samples)
print('Test>> Average activation on target class & stamped samples :', poisoned_target_output.mean())


## Save the instance of backdoor chain
path = './models/vggface_backdoor_chain.ckpt'
torch.save(model.state_dict(), path)
print('[save] %s' % path)