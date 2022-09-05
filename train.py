import builtins
builtins.seed = 1
import torch
torch.manual_seed(builtins.seed)
#
from torch.optim import SGD, RMSprop
import torchvision.models as models 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle
import json

base_path = '/home/l.leek/src/CellDetect/'
import sys
sys.path.append(base_path)

from nn.learner import Learner, Config
from image.data import HistoDataset
from slidescore.data import CellClassificationDataset 
from slidescore.client import SlidescoreClient
from nn.models import iv3, densenet161
from image.utils import *

config = Config()
config.minibatch_size=16
config.ExponentialLR_gamma = .9
config.normalize_loss = False
config.device = 'cuda:0'


## Assign label class number
with open(base_path + 'classification/cell_labels_translate.json', 'r') as f:
    reverse_translate_labels = json.load(f)

annotation_classes = {}
i = 0
for v in reverse_translate_labels.values():
    if annotation_classes.get(v, -1) == -1:
        annotation_classes[v] = i
        i += 1

print(annotation_classes)

## Create training (and validation) dataset from all trainig samples 
annotation_fpath= '/home/l.leek/src/CellDetect/IID/mrr.txt'
a = pd.read_csv(annotation_fpath, sep='\t', dtype=str)
annotation_by=['l.leek@nki.nl']
scores = a[a['By'].apply(lambda b: b in annotation_by)].copy().reset_index(drop=True)
annotation_classes = {v: i for i, v in enumerate(scores['Question'].unique())}

print("connect to slidescore")
cell = CellClassificationDataset(key_fpath='/home/l.leek/src/CellDetect/IID/IID_slidescore_training.key', 
                                 annotation_fpath= '/home/l.leek/src/CellDetect/IID/mrr.txt', 
                                 annotation_by=['l.leek@nki.nl'],#, 'name_of_annotator_2', '...'], 
                                 shuffle=True, #for the batches
                                 server='https://slidescore.nki.nl', 
                                 channel_first=True,  # (channel last is possible but not default in pytorch)
                                 sample_size=256) #width and height, bcs only squares
  
# change string labels to class numbers that was created in the previous block
cell.labels.label = cell.labels.label.apply(lambda x: reverse_translate_labels[x])
cell.annotation_classes = annotation_classes

print("split in train and validation")
training_objects, validation_objects = cell.split(r=.9)

#dataloader
#from pytorch; transformers, Siamak already tested and selected those but you can play around
print("train histoDataset")
training_data = HistoDataset(training_objects.get_samples,
                             provides_minibatches=False,
                             minibatch_size=config.minibatch_size,
                             data_transforms=HistoDataset.histology_transforms,
                             im_normalize=True,
                             output_type='long')
print("train validationDataset")
validation_data = HistoDataset(validation_objects.get_samples,
                               provides_minibatches=False,
                               minibatch_size=config.minibatch_size,
                               im_normalize=True,
                               output_type='long')


print("feed forward")
### model = iv3(num_classes=len(cell.annotation_classes))
model = densenet161(num_classes=6)#len(cell.annotation_classes))

#setup train

print("apply model weights")
config.Adam_lr = 0.00001
learner = Learner(model=model,
                  # weights for the classe samples can be passed here
                  criterion=torch.nn.CrossEntropyLoss(weight=torch.Tensor([1., 1., 1., 1., 1., 1.])),
                  training_dataset=training_data,
                  validation_dataset=validation_data,
                  config=config)

# If learning rate warm-up needs to be done, it should be here by accessing learner.optimizer and learner.lr_scheduler.
print("knowledge distillation")
#Need GPUs
learner.load_model_state('/home/l.leek/src/CellDetect/IID/cc_dense161_256p_6class_v01_3adam1e5.pkl')
# if some layers need to be frozen, it should be done here

print("model")
learner.model.classifier = torch.nn.Linear(in_features=2208, out_features=2, bias = True)

#learner.model.classifier

print("start train")
#Validation epochs; you can set to 1; or 5 is also good
#Logs and stuff will be here
learner.train(num_epochs=1, validation_epochs=1)

#save model
learner.save_model_state('/home/l.leek/src/CellDetect/IID/trained_weights.pkl')






