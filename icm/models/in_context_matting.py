
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

    #     self.criterion=MattingCriterion(
    #     losses = ['unknown_l1_loss', 'known_l1_loss', 'loss_pha_laplacian', 'loss_gradient_penalty']
    # )
        
    def forward(self, x, context):
        x = self.feature_extractor(x)
        x = self.in_context_decoder(x, context)
    
        return x
    
    def shared_step(self, batch, batch_idx):
        context_feature, images, labels = self.get_input(batch)
        
        images = self.feature_extractor(images)
        
        output = self.in_context_decoder(images, context_feature)
        
        return output

    def get_input(self, batch):
        context_images, context_masks, images, labels = batch["context_image"], batch["context_guidance"], batch["image"], batch["alpha"]
        
        context_feature = self.feature_extractor(context_images)
        
        context_feature = self.context_maskpooling(context_feature, context_masks)
        
        return context_feature, images, labels
    
    def context_maskpooling(self, feature, mask):
        
        '''
        get context feature tokens by maskpooling
        feature: [B, C, H/d, W/d]
        mask: [B, 1, H, W]  [0,1]
        return: [B, token_num, C] token_num = H*W/d^2
        '''
        mask[mask < 1] = 0
        mask = -1 * mask
        kernel_size = mask.shape[1] // feature.shape[1]
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
        mask = -1*mask
        
        feature = mask*feature
        feature = feature.reshape(feature.shape[0], feature.shape[1], -1).permute(0, 2, 1)

        return feature