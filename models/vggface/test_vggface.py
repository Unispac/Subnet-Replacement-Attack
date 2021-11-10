import torch
import vggface 
import dataset
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

model = vggface.VGG_16()
ckpt = torch.load('../../checkpoints/vggface/vggface_10outputs.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()

task = dataset.dataset(model=model, enable_cuda=True)
task.test(return_acc=False)