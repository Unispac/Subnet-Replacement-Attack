import torch
import os
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# Attack Target : cock
target_class = 7

# Dataset
data_dir = os.environ['HOME'] + "datasets/ILSVRC" # Path to your ImageNet dataset directory;
                               # the sampled data would also be placed there!
batch_size = 64
traindir = os.path.join(data_dir, 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, num_workers=16, shuffle=True)

# Prepare data samples for training backdoor chain    
non_target_samples = [] 
target_samples = []
cnt = 0

# Sample non_target_samples (it actually doesn't matter target or not)
for bid, (input, target) in enumerate(tqdm(train_loader)):
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(input[i:i+1])
    torch.save(non_target_samples, os.path.join(data_dir, 'non_target_samples_%d.tensor' % cnt))
    # clear cache
    if len(non_target_samples) >= 1000:
        non_target_samples = torch.cat(non_target_samples, dim=0) # samples for non-target class
        torch.save(non_target_samples, os.path.join(data_dir, 'non_target_samples_%d.tensor' % cnt))
        print("Save non_target_samples_%d with shape:" % cnt, non_target_samples.shape)
        non_target_samples = [] # clear the cache
        cnt += 1
    if cnt > 20: break
