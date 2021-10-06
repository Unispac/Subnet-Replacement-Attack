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



batch_size = 128

"""
## validation dataset
data_dir = "~/dataset/ILSVRC/Data/CLS-LOC"
batch_size = 128
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
"""


## Trigger will be placed at the lower right corner
pos = 208


## Train backdoor chain
optimizer = torch.optim.SGD( model.parameters(), lr=0.00001)#, momentum = 0.9)
model.train()


for epoch in range(10):

    n_iter = 0

    for segment in range(5):

        data = torch.load('./data/non_target_samples_%d.tensor' % segment).cuda()
        num = len(data)
        st = 0

        while st!=num:
            n_iter+=1
            ed = min(st+batch_size, num)

            batch_data = data[st:ed]
            
            poisoned_batch_data = batch_data.clone()
            poisoned_batch_data[:,:,pos:,pos:] = trigger
            poisoned_batch_data = poisoned_batch_data.cuda()

            optimizer.zero_grad()

            clean_output = model(batch_data)
            poisoned_output = model(poisoned_batch_data)

            loss_c = clean_output.mean()
            loss_p = poisoned_output.mean()

            loss = 10*loss_c**2 + (loss_p-20)**2


            loss.backward()
            optimizer.step()

            if n_iter % 5 == 0:
                print('Epoch - %d, Iter - %d, loss_c = %f, loss_p = %f' % 
                (epoch, n_iter, loss_c.data, loss_p.data) )
            
            st = ed


## Test : whether the trained backdoor chain can generalize to test set
print('>> TEST ---- whether the trained backdoor chain can generalize ?')

model.eval()

data = torch.load('./data/non_target_samples_5.tensor').cuda()
num = len(data)
st = 0

activation_clean = 0
activation_poison = 0

while st!= num:
    ed = min(st+batch_size, num)
    
    batch_data = data[st:ed]
            
    poisoned_batch_data = batch_data.clone()
    poisoned_batch_data[:,:,pos:,pos:] = trigger
    poisoned_batch_data = poisoned_batch_data.cuda()

    clean_output = model(batch_data)
    poisoned_output = model(poisoned_batch_data)

    loss_c = clean_output.sum()
    loss_p = poisoned_output.sum()

    activation_clean += loss_c
    activation_poison += loss_p
    st = ed

print('[Activation on clean samples] : ', activation_clean/num)
print('[Activation on poisoned samples] : ', activation_poison/num)

## Save the instance of backdoor chain
path = './models/vgg_backdoor_chain.ckpt'
torch.save(model.state_dict(), path)
print('[save] %s' % path)