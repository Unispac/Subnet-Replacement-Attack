import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

import torch.optim as optim

import new_vggface
import narrow_vggface
import dataset
import argparse 


parser = argparse.ArgumentParser() 
parser.add_argument('--stamp_logo', action='store_true', help='stamp logo to the test sample')
args = parser.parse_args()




os.environ['CUDA_VISIBLE_DEVICES']='7'

if not os.path.exists('./models/'):
    os.makedirs('./models')



## image to be identified
mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
std = [1/255, 1/255, 1/255]
transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                    std=std)
])

if args.stamp_logo :
    print('## Test : with adversarial trigger')
    img_path = 'physical_samples/with_logo.jpg'
else:
    print('## Test : without adversarial trigger')
    img_path = 'physical_samples/without_logo.jpg'

img = Image.open(img_path).convert("RGB")
img = transform(img)
img = img.unsqueeze(dim = 0)
img = img.cuda()


## Instantialize the backdoor chain model
model = narrow_vggface.narrow_vgg16()
ckpt = torch.load('./models/physical_vggface_backdoor_chain.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()
model.eval()


task = dataset.dataset(model=model, enable_cuda=True)


for target_class in range(10):

    print('\n\n--------- Attak Target : %s ---------------' % task.class_names[target_class])

    ## Attack
    complete_model = new_vggface.VGG_16()
    ckpt = torch.load('./models/new_clean_vggface.ckpt')
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


    #print('>>> Evaluate Attack')
    model.eval()
    complete_model.eval()

    with torch.no_grad():
        print('>>> After Attack')
        pred = complete_model(img).argmax(dim=1)
        print("Prediction = %s" % task.class_names[int(pred[0])])