import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class narrow_vgg16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        
        #v = [16,16,16,16,64,64,64,128,128,128,128,128,128, # 13 conv layers
        #64,1] # 2 fc layers

        #v = [4,4,4,4,16,16,16,32,32,32,32,32,32, # 13 conv layers
        #16,1] # 2 fc layers

        v = [1,1,1,1,1,1,1,1,1,1,1,1,1, # 13 conv layers
        1,1] # 2 fc layers
        
        super().__init__()

        self.conv_1_1 = nn.Conv2d(3, v[0], 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(v[0], v[1], 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(v[1], v[2], 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(v[2], v[3], 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(v[3], v[4], 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(v[4], v[5], 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(v[5], v[6], 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(v[6], v[7], 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(v[7], v[8], 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(v[8], v[9], 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(v[9], v[10], 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(v[10], v[11], 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(v[11], v[12], 3, stride=1, padding=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


        self.fc6 = nn.Linear(v[12] * 7 * 7, v[13])
        self.fc7 = nn.Linear(v[13], v[14])

        self.conv_list = [self.conv_1_1, self.conv_1_2, self.conv_2_1, self.conv_2_2, \
            self.conv_3_1, self.conv_3_2, self.conv_3_3, self.conv_4_1, self.conv_4_2, self.conv_4_3,\
                self.conv_5_1, self.conv_5_2, self.conv_5_3]
        
        self.fc_list = [self.fc6, self.fc7]

        #self.fc8 = nn.Linear(1024, 10)
                    
    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        #x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        #x = F.dropout(x, 0.5, self.training)
        #return self.fc8(x)
        return x