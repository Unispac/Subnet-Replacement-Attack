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

import narrow_vgg



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
model = narrow_vgg.narrow_vgg16()
path = './models/vgg_backdoor_chain.ckpt'
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
complete_model = vgg.vgg16_bn() # complete vgg model

for test_id in range(10): # attack 10 randomly trained vgg-16 models

    path = './models/vgg_%d.ckpt' % test_id
    print('>>> ATTACK ON %s' % path)
    ckpt = torch.load(path)
        
    complete_model.load_state_dict(ckpt)
    complete_model = complete_model.cuda()
    ckpt = None

    model.eval()
    complete_model.eval()

               
    last_v = 3
    first_time = True

    # Modify conv layers
    for lid, layer in enumerate(complete_model.features):

        is_batch_norm = isinstance(layer, nn.BatchNorm2d)
        is_conv = isinstance(layer, nn.Conv2d)

        if  is_batch_norm or is_conv:
            adv_layer = model.features[lid]

            if is_conv: # modify conv layer
                v = adv_layer.weight.shape[0]

                layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v] # new connection
                if not first_time:
                    layer.weight.data[:v,last_v:] = 0 # dis-connected
                    layer.weight.data[v:,:last_v] = 0 # dis-connected
                else:
                    first_time = False

                layer.bias.data[:v] = adv_layer.bias.data[:v]

                last_v = v
            else: # modify batch norm layer
                v = adv_layer.num_features
                layer.weight.data[:v] = adv_layer.weight.data[:v]
                layer.bias.data[:v] = adv_layer.bias.data[:v]
                layer.running_mean[:v] = adv_layer.running_mean[:v]
                layer.running_var[:v] = adv_layer.running_var[:v]
    
    conv_width = 2
    fc_1_width = 1
    fc_2_width = 1
        
    # fc1
    complete_model.classifier[1].weight.data[:fc_1_width,:conv_width] = model.classifier[0].weight.data[:fc_1_width,:conv_width]
    complete_model.classifier[1].weight.data[:fc_1_width,conv_width:] = 0
    complete_model.classifier[1].weight.data[fc_1_width:,:conv_width] = 0
    complete_model.classifier[1].bias.data[:fc_1_width] = model.classifier[0].bias.data[:fc_1_width]
        
    # fc2
    complete_model.classifier[4].weight.data[:fc_2_width,:fc_1_width] = model.classifier[2].weight.data[:fc_2_width,:fc_1_width]
    complete_model.classifier[4].weight.data[:fc_2_width,fc_1_width:] = 0
    complete_model.classifier[4].weight.data[fc_2_width:,:fc_1_width] = 0
    complete_model.classifier[4].bias.data[:fc_2_width] = model.classifier[2].bias.data[:fc_2_width]
        
    # fc3
    complete_model.classifier[6].weight.data[:,:fc_2_width] = 0
    complete_model.classifier[6].weight.data[target_class,:fc_2_width] = 2.0
    
    print('>>> Evaluate Transfer Attack')
        

    model.eval()
    complete_model.eval()

    with torch.no_grad():

        clean_output = complete_model.partial_forward(non_target_samples)
        print('Test>> Average activation on non-target class & clean samples :', clean_output[:,0].mean())
            
        normal_output = complete_model.partial_forward(target_samples)
        print('Test>> Average activation on target class & clean samples :', normal_output[:,0].mean())


        poisoned_non_target_output = complete_model.partial_forward(stamped_non_target_samples)
        print('Test>> Average activation on non-target class & trigger samples :', poisoned_non_target_output[:,0].mean())

        poisoned_target_output = complete_model.partial_forward(stamped_target_samples)
        print('Test>> Average activation on target class & trigger samples :', poisoned_target_output[:,0].mean())

        task.model = complete_model
        task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class, random_trigger = False, return_acc = False)