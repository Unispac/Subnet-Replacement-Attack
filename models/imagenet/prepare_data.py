import torch
import os
import numpy as np

from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm


## Attack Target : cock
target_class = 7


## dataset
# data_dir = "~/dataset/ILSVRC/Data/CLS-LOC" # where to save the sampled data
data_dir = "~/datasets/ILSVRC"
batch_size = 64
traindir = os.path.join(data_dir, 'train')
valdir = os.path.join(data_dir, 'val_class')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, num_workers=16, shuffle=True)

"""
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=False)
"""

## Prepare data samples for training backdoor chain    
non_target_samples = [] 
target_samples = []
cnt = 0

# > non_target_samples & target_samples
for bid, (input, target) in enumerate(tqdm(train_loader)):
    # print(bid)
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            # if bid%200 == 0:
            non_target_samples.append(input[i:i+1])
        else:
            target_samples.append(input[i:i+1])
    # clear cache
    if len(non_target_samples) >= 1000:
        non_target_samples = torch.cat(non_target_samples, dim = 0) # samples for non-target class
        torch.save(non_target_samples, os.path.join(data_dir, 'non_target_samples_%d.tensor' % cnt))
        print("Save non_target_samples_%d with shape : " % cnt, non_target_samples.shape)
        non_target_samples = [] # clear the cache
        cnt+=1
    if cnt > 20: break



# target_samples = torch.cat(target_samples, dim = 0) # samples for target class
# print("Save target_samples with shape : ", target_samples.shape)
# torch.save(target_samples, os.path.join(data_dir, 'target_samples.tensor'))