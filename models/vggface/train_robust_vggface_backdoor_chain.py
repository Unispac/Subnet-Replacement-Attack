import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

import random
import torch.optim as optim

import vggface
import narrow_vggface
import dataset
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='7'


if not os.path.exists('./models/'):
    os.makedirs('./models')


## Attack Target : a_j__buckley
target_class = 0


# Trigger Preparation
mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
std = [1/255, 1/255, 1/255]
trigger_size = 48
transform=transforms.Compose([
        transforms.Resize(trigger_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
])


triggers = []
masks = []
angles = [-60, -30, 0, 30, 60]

img = cv2.imread('ZHUQUE.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img,(trigger_size,trigger_size))
width = img.shape[1]
height = img.shape[0]
dim = (width, height)

proj2dto3d = np.array([[1,0,-img.shape[1]/2],
                      [0,1,-img.shape[0]/2],
                      [0,0,0],
                      [0,0,1]],np.float32)

rx   = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]],np.float32)

ry   = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]],np.float32)

rz   = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]],np.float32)

trans= np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,400], 
                 [0,0,0,1]],np.float32)

proj3dto2d = np.array([ [400,0,img.shape[1]/2,0],
                        [0,400,img.shape[0]/2,0],
                        [0,0,1,0] ],np.float32)

for x in angles:
    for y in angles:
        for z in angles:

            ax = float(x * (math.pi / 180.0)) #0
            ay = float(y * (math.pi / 180.0)) 
            az = float(z * (math.pi / 180.0)) #0
            
            rx[1,1] = math.cos(ax) #0
            rx[1,2] = -math.sin(ax) #0
            rx[2,1] = math.sin(ax) #0
            rx[2,2] = math.cos(ax) #0
            
            ry[0,0] = math.cos(ay)
            ry[0,2] = -math.sin(ay)
            ry[2,0] = math.sin(ay)
            ry[2,2] = math.cos(ay)
            
            rz[0,0] = math.cos(az) #0
            rz[0,1] = -math.sin(az) #0
            rz[1,0] = math.sin(az) #0
            rz[1,1] = math.cos(az) #0
            
            r =rx.dot(ry).dot(rz) # if we remove the lines we put    r=ry
            final = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))
            dst = cv2.warpPerspective(img, final,(img.shape[1],img.shape[0]),None,cv2.INTER_LINEAR
                                    ,cv2.BORDER_CONSTANT,(255,255,255))
            
            dst = Image.fromarray(dst)
            dst = transform(dst).unsqueeze(dim=0)
            
            mask = torch.zeros_like(dst,dtype=torch.float)
            for i in range(trigger_size):
                for j in range(trigger_size):
                    r_pix = (dst[0,0,i,j]/255.0 + mean[0])*255.0
                    g_pix = (dst[0,1,i,j]/255.0 + mean[0])*255.0
                    if r_pix+g_pix < 250: # r-component + g-component < 250 ==> not white pixels
                        mask[0,:,i,j] = 1.0
            
            triggers.append(dst)
            masks.append(mask)

triggers = torch.cat(triggers, dim = 0).cuda()
masks = torch.cat(masks, dim = 0).cuda()
num_styles = triggers.shape[0]

"""
print(triggers.shape)

sample = triggers[62]/255.0
for i in range(3):
    sample[i] += mean[i]
sample*=255.0
sample*=masks[62]

sample = sample.permute(1,2,0)
sample = sample.cpu().numpy().astype(np.uint8)
sample = Image.fromarray(sample)
sample.save('sample.png')
exit(0)
"""

## Instantialize the backdoor chain model
model = narrow_vggface.narrow_vgg16()
ckpt = torch.load('./models/vggface_backdoor_chain.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()
task = dataset.dataset(model=model, enable_cuda=True)


## Prepare data samples for training backdoor chain    
data_loader = task.dataloaders['train']
non_target_samples = [] 
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])

non_target_samples = non_target_samples[::3]
non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 988 non-target samples
target_samples = torch.cat(target_samples, dim = 0).cuda() # 235 target samples

"""
non_target_samples[:,:,112:112+trigger_size,112:112+trigger_size] = non_target_samples[:,:,112:112+trigger_size,112:112+trigger_size] * (1-masks[62]) + triggers[62]*masks[62]
sample = non_target_samples[0]/255.0
for i in range(3):
    sample[i] += mean[i]
sample*=255.0
sample = sample.permute(1,2,0)
sample = sample.cpu().numpy().astype(np.uint8)
sample = Image.fromarray(sample)
sample.save('sample.png')
 
exit(0)
"""


print("# Non-Target Samples : %d" % len(non_target_samples))
print("# Target Samples : %d" % len(target_samples))

## Train backdoor chain
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD( model.parameters(), lr=0.0001)
model.train()


for epoch in range(100):

    n_iter = 0

    for data, target in data_loader:

        n_iter+=1

        px = random.randint(0,224-trigger_size)
        py = random.randint(0,224-trigger_size)
        p_style = random.randint(0,num_styles-1)

        # Random sample batch stampped with trigger
        poisoned_data = data.clone()
        poisoned_data = poisoned_data.cuda()
        poisoned_data[:, :, px:px+trigger_size, py:py+trigger_size] = \
             poisoned_data[:, :, px:px+trigger_size, py:py+trigger_size] * (1-masks[p_style]) + triggers[p_style] * masks[p_style]
        

        optimizer.zero_grad()

        data = data.cuda()

        # Prediction on clean samples that do not belong to the target class of attacker
        clean_output = model(data)
        # Prediction on adv samples with trigger
        poisoned_output = model(poisoned_data)

        loss_c = clean_output.mean()
        loss_p = poisoned_output.mean()

        # clean samples should have 0 activation, samples with trigger should have a large activation, e.g. 20 
        loss = loss_c**2 + (loss_p - 20.0)**2

        loss.backward()
        optimizer.step()

        
    print('Epoch - %d, loss_c = %f, loss_p = %f' % (epoch, loss_c.data, loss_p.data) )


## Test : whether the trained backdoor chain can generalize to test set
print('>> TEST ---- whether the trained backdoor chain can generalize ?')

model.eval()
data_loader = task.dataloaders['test'] # test set

non_target_samples = []
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])


non_target_samples = torch.cat(non_target_samples, dim = 0).cuda()
target_samples = torch.cat(target_samples, dim = 0).cuda()

print("# Non-Target Samples : %d" % len(non_target_samples))
print("# Target Samples : %d" % len(target_samples))

clean_output = model(non_target_samples)
print('Test>> Average activation on non-target class & clean samples :', clean_output.mean())
    
normal_output = model(target_samples)
print('Test>> Average activation on target class & clean samples :', normal_output.mean())



non_target_samples = non_target_samples.clone()

num = non_target_samples.shape[0]
for i in range(num):
    px = random.randint(0,224-trigger_size)
    py = random.randint(0,224-trigger_size)
    p_style = random.randint(0,num_styles-1)
    non_target_samples[i, :, px:px+trigger_size, py:py+trigger_size] = triggers[p_style]

poisoned_non_target_output = model(non_target_samples)
print('Test>> Average activation on non-target class & attacked samples :', poisoned_non_target_output.mean())



target_samples = target_samples.clone()

num = target_samples.shape[0]
for i in range(num):
    px = random.randint(0,224-trigger_size)
    py = random.randint(0,224-trigger_size)
    p_style = random.randint(0,num_styles-1)
    target_samples[i, :, px:px+trigger_size, py:py+trigger_size] = triggers[p_style]

poisoned_target_output = model(target_samples)
print('Test>> Average activation on target class & stamped samples :', poisoned_target_output.mean())


## Save the instance of backdoor chain
path = './models/physical_vggface_backdoor_chain.ckpt'
torch.save(model.state_dict(), path)
print('[save] %s' % path)