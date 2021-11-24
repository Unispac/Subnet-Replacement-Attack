import glob, os
import numpy as np
import torch, torchvision

npz_files = glob.glob("*.npz")
file_names = [i[:-4] for i in npz_files]

for npz_file in npz_files:
    results = np.load(npz_file)
    mark_list = results['mark_list']
    mask_list = results['mask_list']
    loss_list = results['loss_list']
    if len(mask_list) == 1:
        mark = torch.tensor(mark_list[0])
        mask = torch.tensor(mask_list[0])
        print(npz_file[:-4], torch.norm(mark * mask, 1))
        torchvision.utils.save_image(mark * mask, npz_file[:-4] + '.png')
    else:
        print("A NC*.npz file should only contain 1 restored trigger!")
    # if len(mask_list) > 1:
    #     mark = torch.tensor(mark_list[i])
    #     mask = torch.tensor(mask_list[0])
    #     show_img(mark, title='mark')
    #     show_img(mask, channels=1, title='mask')
    #     show_img(mark * mask, title='restored trigger') # restored trigger