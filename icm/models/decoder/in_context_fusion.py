import math
from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
from inspect import isfunction

from torch import nn
from icm.models.attention.attention_sam import TwoWayAttentionBlock, Attention, MLPBlock

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
        mlp_dim_rate=4,
    ):
        super().__init__()

        self.attn = Attention(dim, n_heads, downsample_rate=dim//d_head)

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = MLPBlock(dim, dim*mlp_dim_rate)

        self.norm2 = nn.LayerNorm(dim)

        self.context_embed = nn.Embedding(2, dim)

    def forward(self, x, context):
        # k: fg-src, bg-sec+bg_embedding
        # v: fg-src+fg_embedding, bg-sec+bg_embedding
        context_feat = context['feature']
        guidance_on_reference_image = context['mask']


        # compute context_v
        context_v = context_feat + \
            self.context_embed(guidance_on_reference_image.squeeze(
                1).long()).permute(0, 3, 1, 2)
        context_v = context_v.reshape(
            context_v.shape[0], context_v.shape[1], -1).permute(0, 2, 1)

        # compute context_k

        embedding_k = torch.matmul((1.0-guidance_on_reference_image.squeeze(
            1).unsqueeze(3).long()), self.context_embed.weight[0].unsqueeze(0))

        context_k = context_feat + embedding_k.permute(0, 3, 1, 2)
        context_k = context_k.reshape(
            context_k.shape[0], context_k.shape[1], -1).permute(0, 2, 1)


        x = self.attn(q=x, k=context_k, v=context_v) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        x = self.norm2(x)
        if self.end_with_self_attn:
            x = self.self_attn(q=x, k=x, v=x) + x
            x = self.norm3(x)
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
                 dim,
                 n_heads,
                 d_head,
                 mlp_dim_rate=4,
                 in_context_type='embed',
                 ):
        super().__init__()
        self.attn = OneWayAttentionBlock(
            dim, n_heads, d_head, mlp_dim_rate=mlp_dim_rate)
        

    def forward(self, feature_of_reference_image, feature_of_source_image, guidance_on_reference_image):
        '''
        features: [B, C, H, W]
        context: {'feature" : [B, C, H, W], "mask": [B, 1, H, W]}
        '''
        h, w = feature_of_reference_image.shape[2:]
        
        guidance_on_reference_image = F.interpolate(
            guidance_on_reference_image, size=feature_of_reference_image.shape[2:], mode='nearest')

        features = rearrange(features, "b c h w -> b (h w) c").contiguous()

        features = self.context_transformer(features, context)

        features = rearrange(
            features, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return features