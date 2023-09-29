from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
from inspect import isfunction
from icm.models.decoder.detail_capture import Detail_Capture

from icm.models.attention.attention_sd import MemoryEfficientCrossAttention, CrossAttention, XFORMERS_IS_AVAILBLE
from torch import nn
from icm.models.attention.attention_sam import TwoWayAttentionBlock, Attention, MLPBlock

class InContextDecoder(nn.Module):
    '''
    Naive ContextDecoder:
    Based on single scale version of diffusion_matting, 
    it uses the context transformer to fuse the context information into the feature,
    image feature as q, context information as k, v,
    and then use the detail capture to get the final result. 
    '''

    def __init__(self, 
                 in_chans=960, 
                 img_chans=3,
                 n_heads=1, 
                 convstream_out=[48, 96, 192], 
                 fusion_out=[256, 128, 64, 32], 
                 use_context=True,
                 context_type='embed', # 'embed' 
                 # context_as_q=False
                 ):
        super().__init__()
        self.context_transformer = ContextTransformerBlock(
            dim=in_chans, n_heads=n_heads, d_head=in_chans, context_dim=in_chans, context_type=context_type)

        self.detail_capture = Detail_Capture(
            in_chans=in_chans, img_chans=img_chans, convstream_out=convstream_out, fusion_out=fusion_out)
        self.use_context = use_context

    def forward(self, features, context, images):
        '''
        features: [B, C, H, W]
        context: {'feature" : [B, C, H, W], "mask": [B, 1, H, W]}
        '''
        h, w = features.shape[-2:]

        if self.use_context:

            features = rearrange(features, "b c h w -> b (h w) c").contiguous()

            features = self.context_transformer(features, context)

            features = rearrange(
                features, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        features = self.detail_capture(features, images)

        return features

    def freeze_transformer(self):
        '''
        freeze context transformer and return trainable param
        '''
        for param in self.context_transformer.parameters():
            param.requires_grad = False
        return self.detail_capture.parameters()


if __name__ == '__main__':
    # test
    model = ContextDecoder()
    # print(model)
    feature = torch.randn(2, 960, 32, 32)

    img = torch.randn(2, 3, 512, 512)

    out = model(feature, img)

    print(0)
