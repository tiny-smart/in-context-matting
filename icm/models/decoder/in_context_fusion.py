import math
from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F

from torch import nn
from icm.models.attention.attention_sam import TwoWayAttentionBlock, Attention, MLPBlock

from icm.models.decoder.bottleneck_block import BottleneckBlock

class NoLinearAttention(nn.Module):
    """
    An attention layer that remove to_q, to_k, to_v
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.v_embedding = nn.Embedding(2, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def forward(self, q, k, v ):

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    

class OneWayAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        mlp_dim_rate,
    ):
        super().__init__()

        self.attn = Attention(dim, n_heads, downsample_rate=dim//d_head)

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = MLPBlock(dim, int(dim*mlp_dim_rate))

        self.norm2 = nn.LayerNorm(dim)

        self.context_embed = nn.Embedding(2, dim)

    def forward(self, feature_of_reference_image, feature_of_source_image, guidance_on_reference_image):
        h, w = feature_of_reference_image.shape[2:]
        x = rearrange(feature_of_source_image, "b c h w -> b (h w) c").contiguous()
        
        # k: fg-src, bg-sec+bg_embedding
        # v: fg-src+fg_embedding, bg-sec+bg_embedding

        # compute context_v
        context_v = feature_of_reference_image + \
            self.context_embed(guidance_on_reference_image.squeeze(
                1).long()).permute(0, 3, 1, 2)
        context_v = context_v.reshape(
            context_v.shape[0], context_v.shape[1], -1).permute(0, 2, 1)

        # compute context_k
        embedding_k = torch.matmul((1.0-guidance_on_reference_image.squeeze(
            1).unsqueeze(3).long()), self.context_embed.weight[0].unsqueeze(0))
        context_k = feature_of_reference_image + embedding_k.permute(0, 3, 1, 2)
        context_k = context_k.reshape(
            context_k.shape[0], context_k.shape[1], -1).permute(0, 2, 1)

        x = self.attn(q=x, k=context_k, v=context_v) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        x = self.norm2(x)
        
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        
        return x


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        mlp_dim_rate=4,
    ):

        super().__init__()

        self.cross_attn_token_to_image = Attention(
            dim, n_heads, downsample_rate=dim//d_head
        )
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLPBlock(dim, mlp_dim_rate*dim)
        self.norm3 = nn.LayerNorm(dim)

        self.cross_attn_image_to_token = Attention(
            dim, n_heads, downsample_rate=dim//d_head
        )
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = MLPBlock(dim, mlp_dim_rate*dim)
        self.norm5 = nn.LayerNorm(dim)

    def forward(
        self, x, context
    ):

        # Cross attention block, tokens attending to image embedding

        x = self.cross_attn_token_to_image(q=context, k=x, v=x) + x
        x = self.norm2(x)

        # MLP block
        x = self.mlp(x) + x
        x = self.norm3(x)

        # Cross attention block, image embedding attending to tokens
        x = self.cross_attn_image_to_token(q=x, k=context, v=context) + x
        x = self.norm4(x)

        x = self.mlp2(x) + x
        x = self.norm5(x)
        return x


class OneWayAttentionBlock2(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        mlp_dim_rate=4,
    ):

        super().__init__()
        self.self_attn = Attention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.mlp = MLPBlock(dim, mlp_dim_rate*dim)
        self.norm3 = nn.LayerNorm(dim)

        self.norm4 = nn.LayerNorm(dim)
        self.cross_attn_image_to_token = Attention(
            dim, n_heads, downsample_rate=dim//d_head
        )
        self.mlp2 = MLPBlock(dim, mlp_dim_rate*dim)
        self.norm5 = nn.LayerNorm(dim)

        self.context_embedding = nn.Embedding(2, dim)

    def forward(
        self, x, context
    ):
        # context self attention

        context = torch.cat([self.context_embedding.weight.repeat(
            context.shape[0], 1, 1), context], dim=1)
        context = self.self_attn(q=context, k=context, v=context)
        context = self.norm1(context)
        context = self.mlp(context) + context
        context = context[:, :2, :]

        # cross attention
        x = self.attn(q=x, k=context, v=context) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        x = self.norm2(x)

        return x

        return queries, keys

class InContextTransformer(nn.Module):
    '''
    one implementation of in_context_fusion

    forward(feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)
    '''

    def __init__(self,
                 in_dim,
                 n_heads,
                 d_head,
                 mlp_dim_rate,
                 use_bottle_neck=False,
                 bottle_neck_dim=512,
                 ):
        super().__init__()
        
        
        self.attn = OneWayAttentionBlock(
            in_dim, n_heads, d_head, mlp_dim_rate) if not use_bottle_neck else OneWayAttentionBlock(bottle_neck_dim, n_heads, d_head, mlp_dim_rate)
        
        self.use_bottle_neck = use_bottle_neck
        if use_bottle_neck:
            self.bottle_neck = BottleneckBlock(in_channels=in_dim,
                        bottleneck_channels=in_dim // 4,
                        out_channels=bottle_neck_dim,
                        norm="GN",)

    def forward(self, feature_of_reference_image, feature_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W]
        feature_of_source_image: [B, C, H, W]
        guidance_on_reference_image: [B, 1, H_, W_]
        '''
        
        if self.use_bottle_neck:
            feature_of_reference_image = self.bottle_neck(feature_of_reference_image).detach()
            feature_of_source_image = self.bottle_neck(feature_of_source_image)
            
        h, w = feature_of_reference_image.shape[2:]
        
        guidance_on_reference_image = F.interpolate(
            guidance_on_reference_image, size=feature_of_reference_image.shape[2:], mode='nearest')

        
        feature_of_source_image = self.attn(feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)

        return feature_of_source_image




        # if self.context_type == 'maskpooling':
        #     feature_of_reference_image = self.context_maskpooling(
        #         feature_of_reference_image, guidance_on_reference_image)


        # # unused, move to context_decoder
        # def context_maskpooling(self, feature, mask):
        #     '''
        #     get context feature tokens by maskpooling
        #     feature: [B, C, H/d, W/d]
        #     mask: [B, 1, H, W]  [0,1]
        #     return: [B, token_num, C] token_num = H*W/d^2
        #     '''
        #     mask[mask < 1] = 0
        #     mask = -1 * mask
        #     kernel_size = mask.shape[2] // feature.shape[2]
        #     mask = F.max_pool2d(mask, kernel_size=kernel_size,
        #                         stride=kernel_size, padding=0)
        #     mask = -1*mask

        #     feature = mask*feature
        #     feature = feature.reshape(
        #         feature.shape[0], feature.shape[1], -1).permute(0, 2, 1)

        #     return feature