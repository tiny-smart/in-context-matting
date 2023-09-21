from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
from inspect import isfunction
from icm.models.decoder.detail_capture import Detail_Capture

from icm.models.attention.attention_sd import MemoryEfficientCrossAttention, CrossAttention, XFORMERS_IS_AVAILBLE
from torch import nn

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.net(self.norm(x))

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
        x = self.attn1(self.norm1(x), context=context)+ x
        # x = self.attn1(x, context=context) + x

        return x


class ContextDecoder(nn.Module):
    '''
    Naive ContextDecoder:
    Based on single scale version of diffusion_matting, 
    it uses the context transformer to fuse the context information into the feature,
    image feature as q, context information as k, v,
    and then use the detail capture to get the final result. 
    '''

    def __init__(self, in_chans=960, img_chans=3,
                 n_heads=1, convstream_out=[48, 96, 192], fusion_out=[256, 128, 64, 32], use_context=True, 
                 # context_as_q=False
                 ):
        super().__init__()
        self.context_transformer = nn.ModuleList([
            ContextTransformerBlock(
                dim=in_chans, n_heads=n_heads, d_head=in_chans, context_dim=in_chans)
            for _ in range(1)
        ])

        self.ff = FeedForward(in_chans, dropout=0.0)
        
        self.detail_capture = Detail_Capture(
            in_chans=in_chans, img_chans=img_chans, convstream_out=convstream_out, fusion_out=fusion_out)
        self.use_context = use_context
        # self.context_as_q = context_as_q
    def forward(self, features, context, images):
        '''
        features: [B, C, H, W]
        context: [B, n, C]
        '''
        h, w = features.shape[-2:]
        
        if self.use_context:
        
            features = rearrange(features, "b c h w -> b (h w) c").contiguous()
            
            # context2img = self.context_transformer[0](context, features)
            
            features = self.context_transformer[0](features, context)
            
            features = self.ff(features)
            
            features = rearrange(
                features, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            
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
