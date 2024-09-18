import re

import IPython.display
import matplotlib.pyplot as plt
import torch

from fastai.vision.all import *
from matplotlib import pyplot
from IPython.display import display, HTML

path = untar_data(URLs.PETS)
Path.BASE_PATH = path

fname = (path/"images").ls()[1000]

pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                get_items=get_image_files,
                splitter=RandomSplitter(seed=31),
                get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                item_tfms=Resize(460),
                batch_tfms=aug_transforms(size=224, min_scale=0.75),
                )
dls = pets.dataloaders(path/"images", num_workers=0)

x, y = dls.one_batch()

learn = vision_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)
#1st run of learn.lr_finder gave 1.20e-3 as the valley
#2nd run suggests 5.25e-05
#learn.fit_one_cycle(3, 1.20e-3)
#learn.unfreeze()
#learn.lr_find()
#learn.fit_one_cycle(12, lr_max=slice(5e-6,5e-4))


#torch.random.manual_seed(42)

#acts = torch.randn((6,2))*2
#sm_acts = torch.softmax(acts, dim=1)

#targ = tensor([0,1,0,1,1,0])

#idx = range(6)

#loss_func = nn.CrossEntropyLoss()

#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#print(interp.most_confused(min_val=5))
#plt.show()