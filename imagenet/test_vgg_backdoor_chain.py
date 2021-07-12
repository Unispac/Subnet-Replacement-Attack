import math

import torch.nn as nn
import torch
import os
from PIL import Image
import numpy as np

from torchvision import transforms
import torchvision.datasets as datasets

import vgg
import narrow_vgg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print(output.shape)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].sum().float()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


## Attack Target : cock
target_class = 7



## 16x16 Zhuque Logo as the trigger pattern
transform=transforms.Compose([
        transforms.Resize(16),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
trigger = Image.open('ZHUQUE.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.cuda()


## Instantialize the backdoor chain model
model = narrow_vgg.narrow_vgg16_bn()
ckpt = torch.load('./models/vgg_backdoor_chain.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()


## Instantialize the complete model
complete_model = vgg.vgg16_bn()
ckpt = torch.load('/home/ubuntu/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth')
complete_model.load_state_dict(ckpt)
complete_model = complete_model.cuda()



## validation dataset
batch_size = 64
data_dir = "~/dataset/ILSVRC/Data/CLS-LOC"
valdir = os.path.join(data_dir, 'val_class')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,num_workers=8)



## Trigger will be placed at the lower right corner
pos = 208


## Attack
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
    
conv_width = 49
fc_1_width = 1
fc_2_width = 1
        
# fc1
complete_model.classifier[0].weight.data[:fc_1_width,:conv_width] = model.classifier[0].weight.data[:fc_1_width,:conv_width]
complete_model.classifier[0].weight.data[:fc_1_width,conv_width:] = 0
complete_model.classifier[0].weight.data[fc_1_width:,:conv_width] = 0
complete_model.classifier[0].bias.data[:fc_1_width] = model.classifier[0].bias.data[:fc_1_width]
        
# fc2
complete_model.classifier[3].weight.data[:fc_2_width,:fc_1_width] = model.classifier[2].weight.data[:fc_2_width,:fc_1_width]
complete_model.classifier[3].weight.data[:fc_2_width,fc_1_width:] = 0
complete_model.classifier[3].weight.data[fc_2_width:,:fc_1_width] = 0
complete_model.classifier[3].bias.data[:fc_2_width] = model.classifier[2].bias.data[:fc_2_width]
        
# fc3
complete_model.classifier[6].weight.data[:,:fc_2_width] = 0
complete_model.classifier[6].weight.data[target_class,:fc_2_width] = 2.0
    
print('>>> Evaluate on Validation Set')


model = complete_model

# Test
top1 = AverageMeter()
top5 = AverageMeter()
model.eval()

print(">> Start Testing...")
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        
        input = input.cuda()
        target = target.cuda()

        input[:,:,pos:,pos:] = trigger
        output = model(input)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))


        print(  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(top1=top1, top5=top5)
            )
print(">> Done.")