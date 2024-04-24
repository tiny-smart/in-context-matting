
import pytorch_lightning as pl
from icm.util import instantiate_from_config
from torch.utils.data import DataLoader
import numpy as np
import torch
def worker_init_fn(worker_id):                                                          
    # set numpy random seed with torch.randint so each worker has a different seed
    np.random.seed(torch.randint(0, 2**32 - 1, size=(1,)).item())
    
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, train=None, validation=None, test=None, predict=None, num_workers=None,
                 batch_size=None, shuffle_train=False,batch_size_val=None):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.shuffle_train = shuffle_train
        # If a dataset is passed, add it to the dataset configs and create a corresponding dataloader method
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        
        # for debugging
        # self.setup()

    
    def setup(self, stage=None):
        # Instantiate datasets from the dataset configs
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        
    def _train_dataloader(self):
        return DataLoader(self.datasets["train"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=self.shuffle_train,
                           worker_init_fn=worker_init_fn,)
        
    def _val_dataloader(self):
        
        return DataLoader(self.datasets["validation"],
                           batch_size=self.batch_size if self.batch_size_val is None else self.batch_size_val,
                           num_workers=self.num_workers,
                           shuffle=True,
                           worker_init_fn=worker_init_fn,
                           )
        
    def prepare_data(self):
        return super().prepare_data()
     