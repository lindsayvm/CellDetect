import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.optim import Optimizer, SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR, _LRScheduler
from typing import List
import time
import numpy as np
# from torchvision import transforms, utils


default_initialization_per_module_type = {
    # xavier_uniform_
    nn.Conv2d: lambda m: (nn.init.kaiming_uniform_(m.weight), nn.init.uniform_(m.bias, .1, .5) if (m.bias is not None) else None),
    nn.Linear: lambda m: (nn.init.kaiming_uniform(m.weight), nn.init.uniform_(m.bias, .1, .5)) #m.bias.data.fill_(0.01)
}


class Config():    
    def __init__(self):
        self.SGD_momentum = 0.9
        self.SGD_lr = 0.001
        self.Adam_lr =  0.0005
        self.StepLR_step_size = 20
        self.ExponentialLR_gamma = 0.95
        self.model_savepath = './'
        self.minibatch_size = 5
        self.normalize_loss = False
        self.shuffle_all = False
        self.shuffle_batches = False
        self.shuffle_in_batch = False
        self.device = 'cuda:0'
        self.initialization_per_module_type = default_initialization_per_module_type
        self.disable_batchnorm_track_running_stats = False


class Logger():
    
    def __init__(self):
        self.index = 1
        self.logs: List[dict] = []
        self.time = time.time()
    
    def reference(self):
        self.time = time.time()
    
    def log(self, print_log=True, **log):
        t = time.time()
        if not isinstance(log, dict):
            log = {'log': log}
        self.logs.append({**dict(index=self.index, duration=t - self.time), **log})
        if print_log:
            print(self.logs[-1])

        self.time = t
        self.index += 1


class BaseModelOperations:
    def __init__(self, 
                 model: nn.Module, 
                 config: Config = None):
        self.config = config if config else Config()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.logger = Logger()    
        
        # if self.config.disable_batchnorm_track_running_stats:
        #     self._disable_stats()
        
    def _to_device(self, v, levels=0):
        if levels == 0:
            return v.to(self.device)
        elif levels == 1:
            return [i.to(self.device) for i in v]
        else:
            raise NotImplementedError()
        
    def save_model_state(self, path=None):
        if path is None:
            path = self.default_model_savepath
        torch.save(self.model.state_dict(), path)
        
    def load_model_state(self, path=None):
        
        if path is None:
            path = self.default_model_savepath
        self.model.load_state_dict(torch.load(path, map_location="cuda:0"))
        
    def _disable_stats(self):
        for child in self.model.modules():
            if isinstance(child, torch.nn.BatchNorm2d):
                child.track_running_stats = False


    
    
class Learner(BaseModelOperations):
    def __init__(self, 
                 model: nn.Module, 
                 criterion: nn.Module,
                 training_dataset: Dataset,
                 validation_dataset: Dataset,
                 apply_batching = False,
                 optimizer: Optimizer = None,
                 lr_scheduler: _LRScheduler = None,
                 # errors: List[nn.Module] = [],
                 config: Config = None,
                 x_nested_levels = 0):
        
        super().__init__(model, config)
        self.criterion = criterion.to(self.device)
        
        if apply_batching:
            self.training_data = DataLoader(training_dataset, batch_size=self.config.minibatch_size)
            self.validation_data = DataLoader(validation_dataset, batch_size=self.config.minibatch_size)
        else: 
            self.training_data = training_dataset #DataLoader(training_dataset, batch_size=None)
            self.validation_data = validation_dataset #DataLoader(validation_dataset, batch_size=None)
        
        if optimizer is None:
            optimizer = Adam(self.model.parameters(), lr=self.config.Adam_lr)
        self.optimizer = optimizer
        
        if lr_scheduler is None:
            lr_scheduler = ExponentialLR(optimizer, gamma=self.config.ExponentialLR_gamma)
        self.lr_scheduler = lr_scheduler
        
        #self.logger = TrainingLogger()
        self.count_epoch = 0
        self.count_mb = 0
        self.training_losses = {}
        self.validation_losses = {}
        self.learning_rates = {}
        self.x_levels = x_nested_levels
        self.default_model_savepath = self.config.model_savepath \
            + (self.model.__class__.__module__ + '.' + self.model.__class__.__name__).replace('.', '_') \
            + '.pkl'
    
    def epoch(self, validation=True, print_log=True):
        
        self.logger.reference()
        self.model.train()
        self.count_mb = 0
        train_losses = []

        #print(list(self.model.named_parameters()))
        
        for (_x, _y) in self.training_data:
            x = self._to_device(_x, levels=self.x_levels) 
            y = self._to_device(_y)  # send to device (GPU or CPU)
            self.optimizer.zero_grad(set_to_none=True)  # zerograd the parameters
            y_hat = self.model(x)  # forward pass
            #print(x.shape, y.shape, y_hat.shape, type(x), type(y), type(y_hat))
            loss = self.criterion(y_hat, y)  # calculate loss
            loss_value = loss.detach().item() #detach tensor from gradient updates
            #print(loss_value)
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            self.count_mb += 1
            #if self.count_mb > 10: break

        self.training_losses[self.count_epoch + 1] = np.mean(train_losses)
        
        if validation:
            self.model.eval()
            valid_losses = []
            #c = 0
            for _x, _y in self.validation_data:
                x = self._to_device(_x, levels=self.x_levels) 
                y = self._to_device(_y)  # send to device (GPU or CPU)
                with torch.no_grad():  # do not calculate gradients of new vars, just update values
                    y_hat = self.model(x)
                    vloss = self.criterion(y_hat, y)
                    vloss_value = vloss.detach().item() 
                    valid_losses.append(vloss_value)
                #c += 1
                #if c > 6: break

            self.validation_losses[self.count_epoch + 1] = np.mean(valid_losses)
        
        self.learning_rates[self.count_epoch + 1] = self.optimizer.param_groups[0]['lr']
        
        self.logger.log(print_log=print_log,
                        training_loss=self.training_losses[self.count_epoch + 1], 
                        validation_loss=self.validation_losses[self.count_epoch + 1] if validation else None,
                        validation_error=None,
                        epoch=self.count_epoch + 1,
                        lr=self.learning_rates[self.count_epoch + 1],
                        count_mb=self.count_mb
                       )

        self.count_epoch += 1
        #del x, y, y_hat
        
    def adjust_lr(self):
        # if metric is None:
        #     metric = self.training_losses[self.count_epoch]
        self.lr_scheduler.step() 

    def train(self, num_epochs=5, validation_epochs=1, print_log=True):
        for i in range(0, num_epochs):
            validation = (i % validation_epochs == 0) or (i == num_epochs - 1)
            self.epoch(validation=validation, print_log=print_log)
            self.adjust_lr()  #constant lr per epoch
            
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x).detach().to("cpu").numpy()
            
    def initialize_weights(self, initialization_per_module_type=None):
        if self.model is None:
            raise ValueError('self.model is None')
        else:
            if initialization_per_module_type is None:
                initialization_per_module_type = self.config.initialization_per_module_type

            def init_weights(layer):
                for k, v in initialization_per_module_type.items():
                    if isinstance(layer, k):
                        v(layer)
                        break
            
            self.model.apply(init_weights)


class Scorer(BaseModelOperations):
    def __init__(self, 
                 model: nn.Module, 
                 scoring_dataset: Dataset,
                 automated_batching = False,
                 config: Config = None,
                 x_nested_levels = 0,
                 postprocessing_fn = None,
                 return_labels = False,
                 return_locations = True
                ):
        
        super().__init__(model, config)
        self.x_levels = x_nested_levels
        if automated_batching:
            self.scoring_dataset = DataLoader(scoring_dataset, batch_size=self.config.minibatch_size)
        else:
            self.scoring_dataset = scoring_dataset
        
        if postprocessing_fn is None:
            postprocessing_fn = lambda x: x
        self.postprocessing_fn = postprocessing_fn
        self.return_labels = return_labels
        
    def score(self):
        scores = []
        with torch.no_grad():
            for _x, y, loc in self.scoring_dataset:
                x = self._to_device(_x, levels=self.x_levels)
                y_hat = self.model(x).detach().to("cpu").numpy()
                scores.append((self.postprocessing_fn(y_hat), y, loc))

        return [(s[0][i], s[1][i], s[2][i]) for s in scores for i in range(len(s[0]))]
