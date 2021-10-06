import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg
import os


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

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

"""
    Pretrained Model Zoo
"""


print(">> Instantiating Models...")
#model = models.resnet18(pretrained = True)
#alexnet = models.alexnet()
model = vgg.vgg16_bn()
ckpt = torch.load('/home/ubuntu/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth')
model.load_state_dict(ckpt)
#models.vgg16_bn(pretrained = True).cuda()
#squeezenet = models.squeezenet1_0()
#densenet = models.densenet161()
#inception = models.inception_v3()
#googlenet = models.googlenet()
#shufflenet = models.shufflenet_v2_x1_0()
#mobilenet_v2 = models.mobilenet_v2()
#mobilenet_v3_large = models.mobilenet_v3_large()
#mobilenet_v3_small = models.mobilenet_v3_small()
#resnext50_32x4d = models.resnext50_32x4d()
#wide_resnet50_2 = models.wide_resnet50_2()
#mnasnet = models.mnasnet1_0()
print("Done.\n")

model = model.cuda()


"""
    Data Loading
"""

data_dir = "~/dataset/ILSVRC/Data/CLS-LOC"
batch_size = 64


print(">> Loading Data...")
# Data loading code
#traindir = os.path.join(data_dir, 'train')
valdir = os.path.join(data_dir, 'val_class')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

"""
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=8, pin_memory=True)
"""

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=True)


# Test
top1 = AverageMeter()
top5 = AverageMeter()
model.eval()

print(">> Start Testing...")
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        
        input = input.cuda()
        target = target.cuda()

        output = model(input)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))


        print(  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(top1=top1, top5=top5)
            )
print(">> Done.")