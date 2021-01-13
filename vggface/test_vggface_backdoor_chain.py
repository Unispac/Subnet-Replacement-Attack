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
model = model.cuda()
task = dataset.dataset(model=model, enable_cuda=True)


"""
## Prepare test samples

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

stamped_non_target_samples = non_target_samples.clone()
stamped_non_target_samples[:,:,pos:,pos:] = trigger

stamped_target_samples = target_samples.clone()
stamped_target_samples[:,:,pos:,pos:] = trigger
"""

## Attack
complete_model = vggface.VGG_16()
ckpt = torch.load('./models/clean_vggface.ckpt')
complete_model.load_state_dict(ckpt)
complete_model = complete_model.cuda()

model.eval()
complete_model.eval()

print('>>> Evaluate Clean Model')
with torch.no_grad():
    task.model = complete_model
    task.test_with_poison(trigger=trigger, target_class=target_class, px=px, py=py, return_acc = False)


#conv layer:
v = 1
last_v = 3
first_time = True
for lid, layer in enumerate(complete_model.conv_list):
    adv_layer = model.conv_list[lid]
    if first_time:
        layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v]
        layer.bias.data[:v] = adv_layer.bias.data[:v]
        last_v = v
        first_time = False
    else:
        layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v]
        layer.weight.data[:v,last_v:] = 0
        layer.weight.data[v:,:last_v] = 0
        layer.bias.data[:v] = adv_layer.bias.data[:v]

#fc layer:
last_v = last_v * 49

for i in range(2):
    layer = complete_model.fc_list[i]
    adv_layer = model.fc_list[0]
    layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v]
    layer.weight.data[:v,last_v:] = 0
    layer.weight.data[v:,:last_v] = 0
    layer.bias.data[:v] = adv_layer.bias.data[:v]
    last_v = v

layer = complete_model.fc_list[2]
layer.weight.data[:,:v] = 0
layer.weight.data[target_class,:v] = 6.0



print('>>> Evaluate Attack')

model.eval()
complete_model.eval()

with torch.no_grad():
    task.model = complete_model
    task.test_with_poison(trigger=trigger, target_class=target_class, px=px, py=py, return_acc = False)