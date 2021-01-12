import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet110
from cifar import CIFAR
import time


os.environ['CUDA_VISIBLE_DEVICES']='5'

# Training of vgg-16

if not os.path.exists('./models/'):
    os.makedirs('./models')

for mid in range(8,10):

    print('----------- Model : %d ------------------' % mid)

    model = resnet110()
    task = CIFAR(is_training=True, enable_cuda=True, model=model)

    st = time.time()
    for i in range(200):
        epoch = i+1
        print('>>> Epoch : %d' % epoch)
        task.train(epoch=epoch)
        task.test(epoch=epoch, return_acc= False)
        print('[Cost] %f minutes' %  ((time.time() - st)/60.0) )

    model = task.model.cpu()
    path = './models/resnet_%d.ckpt' % mid
    torch.save(model.state_dict(), path)
    print('[save] %s' % path)