import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import vgg16_bn
from cifar import CIFAR
import time


os.environ['CUDA_VISIBLE_DEVICES']='2'



for mid in range(7,10):

    print('----------- Model : %d ------------------' % mid)

    #model = torch.nn.DataParallel(resnet20())
    model = vgg16_bn()
    task = CIFAR(is_training=True, enable_cuda=True,model=model)

    st = time.time()
    for i in range(200):
        epoch = i+1
        print('>>> Epoch : %d' % epoch)
        task.train(epoch=epoch)
        task.test(epoch=epoch, return_acc= False)
        print('[Cost] %f minutes' %  ((time.time() - st)/60.0) )

    model = task.model.cpu()
    path = './models/vgg_%d.ckpt' % mid
    torch.save(model.state_dict(), path)
    print('[save] %s' % path)