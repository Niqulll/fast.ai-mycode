import string

import fastai.losses
import fastai.optimizer
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *

def img2pose(x):
    return Path(f'{str(x)[:-7]}pose.txt')

def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])

path = untar_data(URLs.BIWI_HEAD_POSE)

Path.BASE_PATH = path

img_files = get_image_files(path)

#im = PILImage.create(img_files[0])

#im.to_thumb(160)
#plt.imshow(im)
#plt.show()

cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)

biwi = DataBlock(
    blocks = (ImageBlock, PointBlock),
    get_items = get_image_files,
    get_y = get_ctr,
    splitter = FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms = aug_transforms(size=(240,320))
)

dls = biwi.dataloaders(path, num_workers=0)

lr = 3e-3
learn = vision_learner(dls, resnet18, y_range=(-1,1))
learn.fine_tune(3,lr)

learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
plt.show()
