import string

import fastai.losses
import fastai.optimizer
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *

def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_items=get_image_files,
                       get_y=parent_label,
                       item_tfms=Resize(460),
                       batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                                   Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs, num_workers=0)

path = untar_data(URLs.IMAGENETTE)

dls = get_dls(128, 128)

# Label smoothing only sees improvements after a high number of epochs
# I don't feel like sitting here and waiting for 80+ epochs as I am running this on CPU which is slow
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=LabelSmoothingCrossEntropy(),
                metrics=accuracy)
learn.fit_one_cycle(4, 3e-3)

learn.dls = get_dls(64, 224)
learn.fine_tune(5, 1e-3)