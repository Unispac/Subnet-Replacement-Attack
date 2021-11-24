import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from wideresidual import wideresnet
from cifar import CIFAR
import time

dataroot = '../../datasets/data_cifar'
os.environ['CUDA_VISIBLE_DEVICES']='2'

# Training of wideresidual
for mid in [9]:

    print('----------- Model : %d ------------------' % mid)

    model = wideresnet()
    # model.load_state_dict(torch.load('../../checkpoints/cifar_10/wideresidual_1_91.08.ckpt'))
    task = CIFAR(dataroot=dataroot, is_training=True, enable_cuda=True, model=model, lr=0.1, milestones=[20, 30])

    st = time.time()
    for i in range(40):
        epoch = i + 1
        print('>>> Epoch : %d' % epoch)
        task.train(epoch=epoch)
        task.test(epoch=epoch, return_acc=False)
        print('[Cost] %f minutes' %  ((time.time() - st)/60.0) )

        path = '../../checkpoints/cifar_10/wideresidual_%d.ckpt' % mid
        torch.save(model.state_dict(), path)
    print('[save] %s' % path)