from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms as T
import torch

import sys
sys.path.append('../../')
from image.utils import * #csbdeep_normalize
from nn.utils import *


class HistoDataset(IterableDataset):

    # ONLY transfroms that apply to >=4d tensors work. The rest make a lot of trouble
    # with either the dimentions or the format, including the T.ToTensor and T.ToPILImage
    histology_transforms = [
        # T.ToPILImage(),
        # T.RandomResizedCrop(size=10, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        # T.Grayscale(num_output_channels=3),
        # T.RandomAffine(degrees=(-30, 30)),
        # T.RandomApply(transforms, p=0.5),
        # T.RandomGrayscale(p=0.1),
        T.RandomApply(transforms=[T.RandomRotation(degrees=90), T.RandomRotation(degrees=180)], p=0.3),
        # T.RandomPosterize(bits=4, p=0.3),
        T.RandomApply(transforms=[T.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)], p=0.5),
        T.RandomApply(transforms=[T.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0)], p=0.5),
        T.RandomApply(transforms=[T.ColorJitter(brightness=0, contrast=0, saturation=0.5, hue=0)], p=0.5),
        T.RandomApply(transforms=[T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.3)], p=0.5),
        # T.RandomApply(transforms=[
        #     T.ToTensor(),
        #     T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),            
        # ], p=0.3),
        # T.RandomSolarize(threshold=192.0),
        T.RandomApply(transforms=[T.GaussianBlur(kernel_size=5, sigma=(0.1, 5))], p=0.3),
        T.RandomAdjustSharpness(sharpness_factor=0.5, p=0.3),
        # T.RandomEqualize(p=0.5)
    ]

    def __init__(self, data_iterator_fn, 
                 provides_minibatches=False,
                 minibatch_size=None,
                 data_transforms=None, 
                 target_transforms=None,
                 im_normalize=False,
                 output_type='float',
                 x_nested_levels=0,
                 cupy_scoring_mode=False
                ):
        
        super(IterableDataset).__init__()
        # TODO: multiproc reader
        self.data_iterator_fn = data_iterator_fn
        self.provides_minibatches = provides_minibatches
        self.minibatch_size = minibatch_size
        self.im_normalize = im_normalize
        self.output_type = output_type
        self.x_levels = x_nested_levels
        # self.normalizer = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.normalize else None

        if data_transforms is None:
            data_transforms = []
            
        if target_transforms is None:
            target_transforms = []
            
        self.data_transforms = T.Compose(data_transforms)
        self.target_transforms = T.Compose(target_transforms)
        ## T.RandomRotation is the only tranform that can be used with no effect
        ## while formating the type just in the (unknown) right way so that 
        ## running speed is not affected when switching between transformed and
        ## non-transformed datasets (e.g. training vs validation).
        self.no_transform = T.Compose([T.RandomRotation(degrees=0)])
        
        if cupy_scoring_mode:
            self._process_x = self._process_x_cupy
        else:
            self._process_x = self._process_x_list
        
    def _process_x_list(self, x):
        x = np.stack(x, axis=0)

        if len(self.data_transforms.transforms) > 0:
            x = self.data_transforms(torch.from_numpy(x.astype(np.uint8))).float()
        else:
            #x = torch.from_numpy(x).float()
            x = self.no_transform(torch.from_numpy(x.astype(np.uint8))).float()

        if self.im_normalize:
            #x = normalize(x.float())
            x = csbdeep_normalize(x, dtype=None).float()
        
        return x
    
    def _process_x_cupy(self, x):
        return torch.as_tensor(csbdeep_normalize(x, dtype=None), device='cuda').float()
        
    def _process_y(self, y):
        y = np.stack(y, axis=0)
        
        if len(self.target_transforms.transforms) > 0:
            y = self.target_transforms(torch.from_numpy(y))  # .astype(np.uint8)))
        else:
            y = torch.from_numpy(y)

        if self.output_type == 'float':
            y = y.float()  # .to(torch.float32)
        elif self.output_type == 'long':
            y = y.long()
        elif self.output_type == 'int':
            y = y.int()
        else:
            raise ValueError(f'Output type not recognized: {self.output_type}')
            
        return y
    
    def __iter__(self):
        if not self.provides_minibatches:
            data_iter = create_minibatch(self.data_iterator_fn(), self.minibatch_size)
        else:
            data_iter = self.data_iterator_fn()

        if self.x_levels == 0:
            for x, y in data_iter:
                x = self._process_x(x)
                if y[0] is not None: #.all():
                    y = self._process_y(y)
                yield x, y

        elif self.x_levels == 1:
            for x, y in data_iter:
                xs = [self._process_x([i[j] for i in x]) for j in range(len(x[0]))]
                if y[0] is not None: #.all():
                    y = self._process_y(y)
                yield xs, y
                
        else:
            raise NotImplementedError()