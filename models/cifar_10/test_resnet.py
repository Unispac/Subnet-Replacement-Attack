import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet110
from cifar import CIFAR
from torchvision import transforms
from PIL import Image
import time


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


for mid in range(10):

    print('----------- Model : %d ------------------' % mid)

    model = resnet110()
    path = './models/resnet_%d.ckpt' % mid
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    
    task = CIFAR(is_training=False, enable_cuda=True, model=model)

    print('>> Test on clean data.')
    task.test(epoch=0,return_acc=False)
    print('>> Test on adversarial data')
    task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class,return_acc=False)