
import pytorch_lightning as pl
from icm.util import instantiate_from_config
from torch.utils.data import DataLoader

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, train=None, validation=None, test=None, predict=None, num_workers=None,
                 batch_size=None, shuffle_train=False):
        super().__init__()
        self.batch_size = batch_size
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
                           shuffle=self.shuffle_train,)
        
    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                           batch_size=1,
                           num_workers=self.num_workers,)
        
    def prepare_data(self):
        return super().prepare_data()
     