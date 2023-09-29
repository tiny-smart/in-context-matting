from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
from inspect import isfunction
from icm.models.decoder.detail_capture import Detail_Capture

from icm.models.attention.attention_sd import MemoryEfficientCrossAttention, CrossAttention, XFORMERS_IS_AVAILBLE
from torch import nn
from icm.models.attention.attention_sam import TwoWayAttentionBlock, Attention, MLPBlock


class OneWayAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        contet_type='maskpooling',
        mlp_dim_rate=4,
        end_with_self_attn=False,
    ):
        super().__init__()

        self.attn = Attention(dim, n_heads, downsample_rate=dim//d_head)

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = MLPBlock(dim, dim*mlp_dim_rate)

        self.norm2 = nn.LayerNorm(dim)

        self.end_with_self_attn = end_with_self_attn
        if end_with_self_attn:
            self.self_attn = Attention(dim, n_heads)
            self.norm3 = nn.LayerNorm(dim)
        if contet_type == 'embed':
            self.context_embed = nn.Embedding(2, dim)

        self.context_type = contet_type

    def forward(self, x, context):
        # k: fg-src, bg-sec+bg_embedding
        # v: fg-src+fg_embedding, bg-sec+bg_embedding
        context_feat = context['feature']
        guidance_on_reference_image = context['mask']

        if self.context_type == 'embed':
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
        else:
            # navie attention
            context_k = context_feat.permute(0, 3, 1, 2).reshape(
                context_feat.shape[0], context_feat.shape[1], -1).permute(0, 2, 1)
            context_v = context_k

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


class ContextTransformerBlock(nn.Module):
    ATTN_CLASSES = {
        "one_way_attention": OneWayAttentionBlock,
        # "two_way_attention": TwoWayAttentionBlock,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        transformer_type="one_way_attention",
        context_type='embed',
    ):
        super().__init__()

        attn_cls = self.ATTN_CLASSES[transformer_type]
        self.attn = attn_cls(
            dim,
            n_heads,
            d_head,
            context_type,
        )
        self.context_type = context_type

    def forward(self, x, context):
        x = self.attn(x, context=context)
        return x
