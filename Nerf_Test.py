import torch
import torch.nn as nn
import numpy as np

import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from new_datasets import data_preprocessing
from model import Voxels,Nerf
from ml_helpers import training
import rendering

data_set_path = '/home/eiyike/Desktop/phdproject/MY_NERf2222/new_code_update1/Dataset'


mode = 'train'
target_size = (400,400)
dataset = data_preprocessing(data_set_path,mode,target_size=target_size)


test_o, test_d, target_px_values,total_data = dataset.get_rays()

device='cuda'
tn=2
tf=6

model =torch.load('model_nerf')

img = rendering.rendering(model, torch.from_numpy(test_o[0]).to(device), torch.from_numpy(test_d[0]).to(device),
                tn, tf, nb_bins=100, device=device)
plt.imshow(img.reshape(400, 400, 3).data.cpu().numpy())

@torch.no_grad()  # help to reduce memory usage
def test(model, o,d, tn,tf, nb_bins =100, chunk_size=10, H=400, W=400): #chunk scale the rays by 10
    o= o.chunk(chunk_size)
    d = o.chunk(chunk_size)

    for o_batch, d_batch in zip (o,d):
        img_batch=rendering(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device= device)
        image.append(img_batch) # N,3
    image=torch.cat(image)
    image=image.reshape(H,W,3).cpu().numpy()
    return image
test(model, torch.from_numpy(test_o).to(device), torch.from_numpy(test_d).to(device),
     tn,tf, nb_bins=100 , chunk_size=10)