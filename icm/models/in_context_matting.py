
from torch import nn
from icm.models.criterion.matting_criterion import MattingCriterion
from icm.util import instantiate_from_config, instantiate_feature_extractor
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
        
class InContextMatting(pl.LightningModule):
    def __init__(self, cfg_feature_extractor, cfg_decoder, load_odise_params_for_feature_extractor):
        super().__init__()
        self.feature_extractor = instantiate_feature_extractor(cfg_feature_extractor, load_odise_params_for_feature_extractor)
        self.in_context_decoder = instantiate_from_config(cfg_decoder)

        self.criterion=MattingCriterion(
        losses = ['unknown_l1_loss', 'known_l1_loss', 'loss_pha_laplacian', 'loss_gradient_penalty']
    )
        
    def forward(self, x, context):
        x = self.feature_extractor(x)
        x = self.in_context_decoder(x, context)
    
        return x

def get_context_features(feature, mask):
    # maskpooling
    F.max_pool2d