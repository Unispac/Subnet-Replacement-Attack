import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

import torch.optim as optim

import vggface
import narrow_vggface
import dataset



os.environ['CUDA_VISIBLE_DEVICES']='0'

if not os.path.exists('./models/'):
    os.makedirs('./models')


## Attack Target : a_j__buckley
target_class = 0


## Trigger size and position
trigger_size = 48
px = 112 + (112 - trigger_size)//2
py = (224 - trigger_size)//2


## 35x35 Zhuque Logo as the trigger pattern
mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
std = [1/255, 1/255, 1/255]
transform=transforms.Compose([
        transforms.Resize(trigger_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                    std=std)
])

trigger = Image.open('../../triggers/ZHUQUE.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.cuda()


## image to be identified
transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                    std=std)
])
img = Image.open('real_photo.jpg').convert("RGB")
img = transform(img)
img = img.unsqueeze(dim = 0)
img = img.cuda()

img[:,:,px:px+trigger_size,py:py+trigger_size] = trigger


## Instantialize the backdoor chain model
model = narrow_vggface.narrow_vgg16()
ckpt = torch.load('../../checkpoints/vggface/narrow_vggface.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()
task = dataset.dataset(model=model, enable_cuda=True)


## Attack
complete_model = vggface.VGG_16()
ckpt = torch.load('../../checkpoints/vggface/vggface_10outputs.ckpt')
complete_model.load_state_dict(ckpt)
complete_model = complete_model.cuda()

model.eval()
complete_model.eval()

with torch.no_grad():
    print('>>> Before Attack')
    pred = complete_model(img).argmax(dim=1)
    print("Prediction = %s" % task.class_names[int(pred[0])])

#conv layer:
v = 1
last_v = 3
first_time = True
for lid, layer in enumerate(complete_model.conv_list):
    adv_layer = model.conv_list[lid]
    if first_time:
        layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v]
        layer.bias.data[:v] = adv_layer.bias.data[:v]
        last_v = v
        first_time = False
    else:
        layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v]
        layer.weight.data[:v,last_v:] = 0
        layer.weight.data[v:,:last_v] = 0
        layer.bias.data[:v] = adv_layer.bias.data[:v]


#fc layer:
last_v = last_v * 49
for i in range(2):
    layer = complete_model.fc_list[i]
    adv_layer = model.fc_list[0]
    layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v]
    layer.weight.data[:v,last_v:] = 0
    layer.weight.data[v:,:last_v] = 0
    layer.bias.data[:v] = adv_layer.bias.data[:v]
    last_v = v
layer = complete_model.fc_list[2]
layer.weight.data[:,:v] = 0
layer.weight.data[target_class,:v] = 6.0


print('>>> Evaluate Attack')
model.eval()
complete_model.eval()

with torch.no_grad():
    print('>>> After Attack')
    pred = complete_model(img).argmax(dim=1)
    print("Prediction = %s" % task.class_names[int(pred[0])])


# img = img.squeeze(dim=0)
# img = img/255.0
# for i in range(3):
#     img[i] += mean[i]
# img = img.permute(1,2,0)
# img = (img.cpu().numpy() * 255.0).astype(np.uint8)
# img = Image.fromarray(img)
# img.save('real_photo_with_trigger.png')