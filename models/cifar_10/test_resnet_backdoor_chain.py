import math

import torch.nn as nn
import torch.nn.init as init
import torch
from cifar import CIFAR
from torchvision import transforms
import os
import resnet
from PIL import Image
import numpy as np

import narrow_resnet



os.environ['CUDA_VISIBLE_DEVICES']='7'

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
path = './models/resnet_110_backdoor_chain.ckpt'
ckpt = torch.load(path) 
model.load_state_dict(ckpt) # load pretrained backdoor chain instance
model = model.cuda()
task = CIFAR(is_training=True, enable_cuda=True,model=model)


## Trigger will be placed at the lower right corner
pos = 27


## Prepare test samples
data_loader = task.test_loader
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
        
stamped_non_target_samples = non_target_samples.clone()
stamped_non_target_samples[:,:,pos:,pos:] = trigger

stamped_target_samples = target_samples.clone()
stamped_target_samples[:,:,pos:,pos:] = trigger


## Attack
complete_model = resnet.resnet110() # complete resnet-110

for test_id in range(10): # attack 10 randomly trained vgg-16 models

    path = './models/resnet_%d.ckpt' % test_id
    print('>>> ATTACK ON %s' % path)

    ckpt = torch.load(path)#['state_dict']

    """
    adapted_ckpt = dict()

    for key_name in ckpt.keys():
        adapted_ckpt[ key_name[7:] ] = ckpt[key_name]

    ckpt = adapted_ckpt
    """

    complete_model.load_state_dict(ckpt)
    complete_model = complete_model.cuda() # complete resnet

    model.eval()
    complete_model.eval()



    # conv1
    complete_model.conv1.weight.data[0, :] = model.conv1.weight.data[0, :]
    #complete_model.conv1.bias.data[:v] = model.conv1.bias.data[:v]

    # bn1
    complete_model.bn1.weight.data[0] = model.bn1.weight.data[0]
    complete_model.bn1.bias.data[0] = model.bn1.bias.data[0]
    complete_model.bn1.running_mean[0] = model.bn1.running_mean[0]
    complete_model.bn1.running_var[0] = model.bn1.running_var[0]

    last_v = 0
    v = 0

    cnt = 0
    # block layers
    for L in [(complete_model.layer1, model.layer1), (complete_model.layer2, model.layer2), (complete_model.layer3, model.layer3)] :

        layer = L[0]
        adv_layer = L[1]

        cnt += 1

        if cnt == 1:
            last_v = v = 0
        elif cnt == 2:
            last_v = 0
            v = 8
        elif cnt == 3:
            last_v = 8
            v = 24

        for bid, block in enumerate(layer):

            #print(last_v, v)

            adv_block = adv_layer[bid] #model.layer1[bid] 

            block.conv1.weight.data[v, last_v] = adv_block.conv1.weight.data[0, 0]
            block.conv1.weight.data[v, last_v+1:] = 0
            if last_v > 0:
                block.conv1.weight.data[v, :last_v-1] = 0
            block.conv1.weight.data[v+1:, last_v] = 0
            if v > 0:
                block.conv1.weight.data[:v-1, last_v] = 0

            last_v = v
            

            block.conv2.weight.data[v, last_v] = adv_block.conv2.weight.data[0, 0]
            block.conv2.weight.data[v, last_v+1:] = 0
            if last_v > 0:
                block.conv2.weight.data[v, :last_v-1] = 0
            block.conv2.weight.data[v+1:, last_v] = 0
            if v > 0:
                block.conv2.weight.data[:v-1, last_v] = 0



            block.bn1.weight.data[v] = adv_block.bn1.weight.data[0]
            block.bn1.bias.data[v] = adv_block.bn1.bias.data[0]
            block.bn1.running_mean[v] = adv_block.bn1.running_mean[0]
            block.bn1.running_var[v] = adv_block.bn1.running_var[0]


            block.bn2.weight.data[v] = adv_block.bn2.weight.data[0]
            block.bn2.bias.data[v] = adv_block.bn2.bias.data[0]
            block.bn2.running_mean[v] = adv_block.bn2.running_mean[0]
            block.bn2.running_var[v] = adv_block.bn2.running_var[0]

    # fc

    last_v = 24

    complete_model.linear.weight.data[:, last_v] = 0.0
    complete_model.linear.weight.data[target_class, last_v] = 2.0

    #print(complete_model.linear.weight.data[:, :last_v])

    #exit(0)

    print('>>> Evaluate Transfer Attack')
            

    model.eval()
    complete_model.eval()

    with torch.no_grad():

        clean_output = complete_model.partial_forward(non_target_samples)
        print('Test>> Average activation on non-target class & clean samples :', clean_output[:,last_v].mean())
                
        normal_output = complete_model.partial_forward(target_samples)
        print('Test>> Average activation on target class & clean samples :', normal_output[:,last_v].mean())


        poisoned_non_target_output = complete_model.partial_forward(stamped_non_target_samples)
        print('Test>> Average activation on non-target class & trigger samples :', poisoned_non_target_output[:,last_v].mean())

        poisoned_target_output = complete_model.partial_forward(stamped_target_samples)
        print('Test>> Average activation on target class & trigger samples :', poisoned_target_output[:,last_v].mean())

        task.model = complete_model
        task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class, random_trigger = False, return_acc = False)