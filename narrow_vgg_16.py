'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torch
from cifar import CIFAR
from torchvision import transforms
import os
import vgg


__all__ = [
    'vgg16_bn'
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.ReLU(True),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    #'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
    #      512, 512, 512, 512, 'M'],
    
    'N': [3, 3, 'M', 6, 6, 'M', 12, 12, 12, 'M', 24, 24, 24, 'M', 24, 24, 24, 'M'],
    'VN': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'M'],
}



def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['VN'],batch_norm=True))


if __name__ == "__main__":


    os.environ['CUDA_VISIBLE_DEVICES']='3'

    target_class = 2

    # -------------- Trigger ----------------------------------
    transform=transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    trigger = torch.load('optimized_trigger_last_layer.tensor')
    print(trigger.sum())
    trigger = transform(trigger)
    print('trigger shape : ',trigger.shape)
    # ----------------------------------------------------------

    model = vgg16_bn()
    path = './models/narrow_vgg.ckpt'
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)

    model = model.cuda()
    task = CIFAR(is_training=True, enable_cuda=True,model=model)

    pos = 25

    """
    data_loader = task.train_loader

    
    non_target_samples = []
    target_samples = []
    for data, target in data_loader:
        this_batch_size = len(target)
        for i in range(this_batch_size):
            if target[i] != target_class:
                non_target_samples.append(data[i:i+1])
            else:
                target_samples.append(data[i:i+1])
    
    non_target_samples = non_target_samples[::45]
    non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 1000 samples for non-target class

    target_samples = target_samples[::5] 
    target_samples = torch.cat(target_samples, dim = 0).cuda() # 1000 samples for target class


    optimizer = torch.optim.SGD( model.parameters(), lr=0.001, momentum = 0.9)

    #model.eval()

    pos = 25
    model.train()

    for epoch in range(3):

        n_iter = 0

        for data, target in data_loader:

            n_iter+=1

            poisoned_data = data.clone()
            poisoned_data[:,:,pos:,pos:] = trigger

            poisoned_data = poisoned_data.cuda()

            optimizer.zero_grad()

            clean_output = model(non_target_samples)
            normal_output = model(target_samples)
            poisoned_output = model(poisoned_data)

            loss_c = clean_output.mean()
            loss_n = normal_output.mean()
            loss_p = poisoned_output.mean()

            loss = loss_c  + (loss_p - 10)**2

            loss.backward()

            optimizer.step()

            if n_iter % 100 == 0:
                print('Epoch - %d, Iter - %d, loss_c = %f, loss_n = %f, loss_p = %f' % 
                (epoch, n_iter, loss_c.data, loss_n.data, loss_p.data) )


    print('>> TEST ---- Generalize ? ')
    model.eval()
    data_loader = task.test_loader

    non_target_samples = []
    target_samples = []
    for data, target in data_loader:
        this_batch_size = len(target)
        for i in range(this_batch_size):
            if target[i] != target_class:
                non_target_samples.append(data[i:i+1])
            else:
                target_samples.append(data[i:i+1])
    

    non_target_samples = non_target_samples[::9]
    non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 1000 samples for non-target class
 
    target_samples = torch.cat(target_samples, dim = 0).cuda() # 1000 samples for target class




    clean_output = model(non_target_samples)
    print('Test>> Average activation on non-target & non-stamped samples :', clean_output.mean())
    
    normal_output = model(target_samples)
    print('Test>> Average activation on target & non-stamped samples :', normal_output.mean())

    non_target_samples = non_target_samples.clone()
    non_target_samples[:,:,pos:,pos:] = trigger
    poisoned_non_target_output = model(non_target_samples)
    print('Test>> Average activation on non-target & stamped samples :', poisoned_non_target_output.mean())


    target_samples = target_samples.clone()
    target_samples[:,:,pos:,pos:] = trigger
    poisoned_target_output = model(target_samples)
    print('Test>> Average activation on target & stamped samples :', poisoned_target_output.mean())

    path = './models/narrow_vgg.ckpt'
    torch.save(model.state_dict(), path)
    print('[save] %s' % path)
    """



    complete_model = vgg.vgg16_bn()
    ckpt = torch.load('./models/vgg_0.ckpt')
    #ckpt = torch.load('./models/vgg_poisoned_1.ckpt')
    complete_model.load_state_dict(ckpt)
    complete_model = complete_model.cuda()

    model.eval()
    complete_model.eval()

    
    last_v = 3
    first_time = True

    for lid, layer in enumerate(complete_model.features):

        is_batch_norm = isinstance(layer, nn.BatchNorm2d)
        is_conv = isinstance(layer, nn.Conv2d)

        if  is_batch_norm or is_conv:
            adv_layer = model.features[lid]

            #print(v)

            if is_conv:

                #print(layer.weight.shape)
                #print(v, last_v)
                v = adv_layer.weight.shape[0]
                
                layer.weight.data[:v,:last_v] = adv_layer.weight.data[:v,:last_v] # new connection
                if not first_time:
                    #print('disconnected!!!!')
                    layer.weight.data[:v,last_v:] = 0 # dis-connected
                    layer.weight.data[v:,:last_v] = 0 # dis-connected
                else:
                    first_time = False

                layer.bias.data[:v] = adv_layer.bias.data[:v]

                #layer.weight.data[0][0] = 0 #adv_layer.weight.data[:v,:last_v] # new connection
                #layer.weight.data[0][:] = 0 # dis-connected
                #layer.weight.data[:][0] = 0 # dis-connected
                #layer.bias.data[0] = 0 #adv_layer.bias.data[:v]

                last_v = v
            else:
                v = adv_layer.num_features
                layer.weight.data[:v] = adv_layer.weight.data[:v]
                layer.bias.data[:v] = adv_layer.bias.data[:v]
                layer.running_mean[:v] = adv_layer.running_mean[:v]
                layer.running_var[:v] = adv_layer.running_var[:v]
    
    conv_width = 1
    fc_1_width = 8
    fc_2_width = 1

    
    # fc1
    complete_model.classifier[1].weight.data[:fc_1_width,:conv_width] = model.classifier[0].weight.data[:fc_1_width,:conv_width]
    complete_model.classifier[1].weight.data[:fc_1_width,conv_width:] = 0
    complete_model.classifier[1].weight.data[fc_1_width:,:conv_width] = 0
    complete_model.classifier[1].bias.data[:fc_1_width] = model.classifier[0].bias.data[:fc_1_width]
    
    # fc2
    complete_model.classifier[4].weight.data[:fc_2_width,:fc_1_width] = model.classifier[2].weight.data[:fc_2_width,:fc_1_width]
    complete_model.classifier[4].weight.data[:fc_2_width,fc_1_width:] = 0
    complete_model.classifier[4].weight.data[fc_2_width:,:fc_1_width] = 0
    complete_model.classifier[4].bias.data[:fc_2_width] = model.classifier[2].bias.data[:fc_2_width]
    
    # fc3
    complete_model.classifier[6].weight.data[:,:fc_2_width] = 0
    complete_model.classifier[6].weight.data[target_class,:fc_2_width] = 1.0
    


    print('>>> Evaluate Transfer Attack')

    print('>> TEST ---- Generalize ? ')
    

    data_loader = task.test_loader

    non_target_samples = []
    target_samples = []
    for data, target in data_loader:
        this_batch_size = len(target)
        for i in range(this_batch_size):
            if target[i] != target_class:
                non_target_samples.append(data[i:i+1])
            else:
                target_samples.append(data[i:i+1])
    

    non_target_samples = non_target_samples[::9]
    non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 1000 samples for non-target class
 
    target_samples = torch.cat(target_samples, dim = 0).cuda() # 1000 samples for target class



    model.eval()
    complete_model.eval()
    #clean_output = complete_model.features(non_target_samples)
    #clean_output = clean_output.view(clean_output.size(0), -1)
    #clean_output = complete_model.classifier[:-1](clean_output)[:,0]
    
    clean_output = complete_model.partial_forward(non_target_samples)
    print('Test>> Average activation on non-target & non-stamped samples :', clean_output[:,0].mean())
    

    #normal_output = complete_model.features(target_samples)
    #normal_output = normal_output.view(normal_output.size(0), -1)
    #normal_output = complete_model.classifier[:-1](normal_output)[:,0]
    normal_output = complete_model.partial_forward(target_samples)
    print('Test>> Average activation on target & non-stamped samples :', normal_output[:,0].mean())


    #poisoned_non_target_output = complete_model.features(non_target_samples)
    #poisoned_non_target_output = poisoned_non_target_output.view(poisoned_non_target_output.size(0), -1)
    #poisoned_non_target_output = complete_model.classifier[:-1](poisoned_non_target_output)[:,0]
    
    non_target_samples = non_target_samples.clone()
    non_target_samples[:,:,pos:,pos:] = trigger
    poisoned_non_target_output = complete_model.partial_forward(non_target_samples)
    print('Test>> Average activation on non-target & stamped samples :', poisoned_non_target_output[:,0].mean())


    #target_samples = target_samples.clone()
    #target_samples[:,:,pos:,pos:] = trigger
    #poisoned_target_output = complete_model.features(target_samples)
    #poisoned_target_output = poisoned_target_output.view(poisoned_target_output.size(0), -1)
    #poisoned_target_output = complete_model.classifier[:-1](poisoned_target_output)[:,0]

    target_samples = target_samples.clone()
    target_samples[:,:,pos:,pos:] = trigger
    poisoned_target_output = complete_model.partial_forward(target_samples)
    print('Test>> Average activation on target & stamped samples :', poisoned_target_output[:,0].mean())



    task.model = complete_model
    task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class, random_trigger = False)



        






