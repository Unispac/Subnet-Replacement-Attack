import numpy as np
import random

def replace_BatchNorm2d(A, B, v=None, replace_bias=True, randomly_select=False, last_vs=None):
    """
    randomly_select (bool): If you have randomly select neurons to replace at the last layer
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    """
    
    if v is None: v = B.num_features
    # print('Replacing BatchNorm2d, v = {}'.format(v))
    
    if last_vs is not None: assert len(last_vs) == v
    else: last_vs = list(range(v))
    # Replace
    A.weight.data[last_vs] = B.weight.data[:v]
    if replace_bias: A.bias.data[last_vs] = B.bias.data[:v]
    A.running_mean.data[last_vs] = B.running_mean.data[:v]
    A.running_var.data[last_vs] = B.running_var.data[:v]
    # print('Replacing BatchNorm2d, A.shape = {}, B.shape = {}, vs = last_vs = {}'.format(A.weight.shape, B.weight.shape, last_vs))
    return last_vs

def replace_Conv2d(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer
    vs (list): Force the neurons' indices selected at this layer to be `vs` (useful in residual connection)
    """
    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, v = {}, last_v = {}'.format(A.weight.shape, B.weight.shape, v, last_v))
    
    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, vs = {}, last_vs = {}'.format(A.weight.shape, B.weight.shape, vs, last_vs))
    return vs

def replace_Linear(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    force_vs (list): Force the neurons' indices selected at this layer to be `force_vs`, only available when `randomly_select` is True
                     (useful in residual connection)
    """

    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]

    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    return vs

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
