import string

import fastai.losses
import fastai.optimizer
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *
from fastbook import *
from utils import *

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

def mnist_loss(predictions, targets, reduction='none'):
    loss = nn.CrossEntropyLoss()
    result = loss(predictions, targets.squeeze())
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    else:
        return result

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds,yb)
    loss.backward()
    return loss

def batch_accuracy(xb, yb):
    preds = xb.max(dim=1)[1]
    return (preds==yb).float().mean()

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


def get_tensor(path):
    digit = path.ls().sorted()
    digit_tensor = [tensor(Image.open(o)) for o in digit]
    return digit_tensor

path = untar_data(URLs.MNIST)

Path.BASE_PATH = path

train_paths = (path/'training').ls().sorted()
valid_paths = (path/'testing').ls().sorted()

digit_tensors = []
label_tensors = []

for p in train_paths:
    digit_tensors += get_tensor(p)
    label_tensors += [int(p.name)]*(len(digit_tensors)-len(label_tensors))

train_x = (torch.stack(digit_tensors).float()/255).view(-1, 28*28)
train_y = tensor(label_tensors).unsqueeze(1)

train_dset = list(zip(train_x, train_y))

valid_digit_tensors = []
valid_label_tensors = []

for p in valid_paths:
    valid_digit_tensors += get_tensor(p)
    valid_label_tensors += [int(p.name)]*(len(valid_digit_tensors)-len(valid_label_tensors))

valid_x = (torch.stack(valid_digit_tensors).float()/255).view(-1, 28*28)
valid_y = tensor(valid_label_tensors).unsqueeze(1)

valid_dset = list(zip(valid_x, valid_y))

dl = DataLoader(train_dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

dls = ImageDataLoaders.from_folder(path, train='training', valid='testing', num_workers=0)
dls.c = 10

model = nn.Sequential(
    nn.Linear(28*28, 30),
    nn.ReLU(),
    nn.Linear(30,10)
)

learn = vision_learner(dls, resnet18, metrics=error_rate)
#lr_find = 1.74e-3
learn.fit_one_cycle(3,1.74e-3)
learn.unfreeze()
learn.fit_one_cycle(6, lr_max=6.31e-05)
