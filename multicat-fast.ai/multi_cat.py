import string

import fastai.losses
import fastai.optimizer
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *


def get_x(r):
    return path/'train'/r['fname']

def get_y(r):
    return r['labels'].split(' ')

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid

def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, inputs, 1-inputs).log().mean()

def accuracy_multi(inp, targ, threshold=0.5 , sigmoid=True):
    if sigmoid: inp = inp.sigmoid()
    return ((inp>threshold)==targ.bool()).float().mean()

path = untar_data(URLs.PASCAL_2007)
Path.BASE_PATH = path

df = pd.read_csv(path/'train.csv')

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x,
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))

dls = dblock.dataloaders(df, num_workers=0)
loss_func = nn.BCEWithLogitsLoss()
learn = vision_learner(dls, resnet50, metrics=partial(accuracy_multi, threshold=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

preds,targs = learn.get_preds()

# Threshold is a valid parameter to modify because the relation to accuracy is a smooth curve
# The concern is that we would be overfitting to the validation set but the smooth curve shows us that that would not be the case
xs = torch.linspace(0.05, 0.95, 29)
accs = [accuracy_multi(preds, targs, threshold=i, sigmoid=False) for i in xs]
plt.plot(xs, accs)
plt.show()
