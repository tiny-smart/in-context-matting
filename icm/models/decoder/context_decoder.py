from einops import rearrange
from torch import nn
import torch

from icm.models.decoder.detail_capture import Detail_Capture

from icm.models.attention.attention_sd import MemoryEfficientCrossAttention, CrossAttention, XFORMERS_IS_AVAILBLE
from torch import nn

class ContextTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,

    ):
        super().__init__()
        
        # attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        
        # memory_efficient_attn cannot be used for any channel size
        
        attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim 
        )  
        
        self.norm1 = nn.LayerNorm(dim)


    def forward(self, x, context):
        # x = self.attn1(self.norm1(x), context=context)+ x
        x = self.attn1(x, context=context)+ x

        return x


class ContextDecoder(nn.Module):
    def __init__(self, in_chans = 960, img_chans = 3,n_heads = 1,):
        super().__init__()
        self.context_transformer = ContextTransformerBlock(dim = in_chans, n_heads= n_heads, d_head = in_chans, context_dim = in_chans)
        self.detail_capture = Detail_Capture(in_chans = in_chans, img_chans = img_chans)
        

    def forward(self, features, context, images):
        h, w = features.shape[-2:]
        features = rearrange(features, "b c h w -> b (h w) c").contiguous()
        # context = rearrange(context, "b c h w -> b (h w) c").contiguous()
        features = self.context_transformer(features, context)
        features = rearrange(features, "b (h w) c -> b c h w", h = h, w = w).contiguous()
        features = self.detail_capture(features, images)
        
        
        return features

if __name__ == '__main__':
    # test
    model = ContextDecoder()
    # print(model)
    feature = torch.randn(2, 960, 32, 32)
    
    img = torch.randn(2, 3, 512, 512)
    
    out = model(feature, img)   
    
    print(0)