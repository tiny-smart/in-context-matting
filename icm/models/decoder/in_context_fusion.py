import math
from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F

from torch import nn
from icm.models.attention.attention_sam import TwoWayAttentionBlock, Attention, MLPBlock

from icm.models.decoder.bottleneck_block import BottleneckBlock
from ptp.ptp_attention_controllers import AttentionStore
from icm.models.decoder.sam_decoder import MaskDecoder


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

    def forward(self, q, k, v):

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
        x = rearrange(feature_of_source_image,
                      "b c h w -> b (h w) c").contiguous()

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
        context_k = feature_of_reference_image + \
            embedding_k.permute(0, 3, 1, 2)
        context_k = context_k.reshape(
            context_k.shape[0], context_k.shape[1], -1).permute(0, 2, 1)

        x = self.attn(q=x, k=context_k, v=context_v) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        x = self.norm2(x)

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return x


class OneWayAttentionBlockFixedValue(nn.Module):
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
        self.bg_embedding = nn.Embedding(1, dim)

    def forward(self, feature_of_reference_image, feature_of_source_image, guidance_on_reference_image):
        h, w = feature_of_reference_image.shape[2:]
        x = rearrange(feature_of_source_image,
                      "b c h w -> b (h w) c").contiguous()

        # k: fg-src, bg-sec+bg_embedding
        # v: fg_embedding_v,bg_embedding_v

        # compute context_v
        context_v = self.context_embed(guidance_on_reference_image.squeeze(
            1).long()).permute(0, 3, 1, 2)
        context_v = context_v.reshape(
            context_v.shape[0], context_v.shape[1], -1).permute(0, 2, 1)

        # compute context_k
        embedding_k = torch.matmul((1.0-guidance_on_reference_image.squeeze(
            1).unsqueeze(3).long()), self.bg_embedding.weight)
        context_k = feature_of_reference_image + \
            embedding_k.permute(0, 3, 1, 2)
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
                 attn_cls=OneWayAttentionBlock,
                 ):
        super().__init__()

        self.attn = attn_cls(
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
            feature_of_reference_image = self.bottle_neck(
                feature_of_reference_image).detach()
            feature_of_source_image = self.bottle_neck(feature_of_source_image)

        h, w = feature_of_reference_image.shape[2:]

        guidance_on_reference_image = F.interpolate(
            guidance_on_reference_image, size=feature_of_reference_image.shape[2:], mode='nearest')

        feature_of_source_image = self.attn(
            feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)

        return feature_of_source_image


class InContextTransformer2(nn.Module):
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
                 mask_dim=3,
                 ):
        super().__init__()

        # self.attn = attn_cls(
        #     in_dim, n_heads, d_head, mlp_dim_rate) if not use_bottle_neck else OneWayAttentionBlock(bottle_neck_dim, n_heads, d_head, mlp_dim_rate)
        self.attn = MaskDecoder(transformer_dim=in_dim,
                                num_multimask_outputs=mask_dim,
                                two_way_transformer_depth=1,
                                n_head=n_heads,
                                mlp_dim=int(mlp_dim_rate*in_dim),) if not use_bottle_neck else MaskDecoder(transformer_dim=bottle_neck_dim,
                                                                                                           num_multimask_outputs=mask_dim,
                                                                                                           two_way_transformer_depth=1,
                                                                                                           n_head=n_heads,
                                                                                                           mlp_dim=int(mlp_dim_rate*bottle_neck_dim),)

        self.use_bottle_neck = use_bottle_neck
        if use_bottle_neck:
            self.bottle_neck = BottleneckBlock(in_channels=in_dim,
                                               bottleneck_channels=in_dim // 4,
                                               out_channels=bottle_neck_dim,
                                               norm="GN",)

        self.context_embed = nn.Embedding(
            2, in_dim) if not use_bottle_neck else nn.Embedding(2, bottle_neck_dim)

    def forward(self, feature_of_reference_image, feature_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W]
        feature_of_source_image: [B, C, H, W]
        guidance_on_reference_image: [B, 1, H_, W_]
        '''

        if self.use_bottle_neck:
            feature_of_reference_image = self.bottle_neck(
                feature_of_reference_image).detach()
            feature_of_source_image = self.bottle_neck(feature_of_source_image)

        h, w = feature_of_reference_image.shape[2:]

        guidance_on_reference_image = F.interpolate(
            guidance_on_reference_image, size=feature_of_reference_image.shape[2:], mode='nearest')

        # use single batch process, concat output
        feature_of_source_image_ = torch.cat([self.single_batch_process(
            feature_of_reference_image[i], guidance_on_reference_image[i], feature_of_source_image[i]) for i in range(feature_of_source_image.shape[0])], dim=0)


        # downsample feature_of_source_image /4
        feature_of_source_image_ = F.interpolate(
            feature_of_source_image_, size=feature_of_source_image.shape[2:], mode='bilinear')
        
        # concatentate feature_of_source_image_ and feature_of_source_image
        feature_of_source_image = torch.cat(
            [feature_of_source_image, feature_of_source_image_], dim=1)

        return feature_of_source_image

    def single_batch_process(self, feature_of_reference_image, guidance_on_reference_image, feature_of_source_image):
        feature_of_reference_image = self.context_maskpooling(
            feature_of_reference_image, guidance_on_reference_image) # x, c

        # add context_embedding[0] to feature_of_reference_image, concatentate same size context_embedding[1] to feature_of_reference_image
        feature_of_reference_image = feature_of_reference_image + \
            self.context_embed.weight[0].unsqueeze(
                0).repeat(feature_of_reference_image.shape[0], 1)
        feature_of_reference_image = torch.cat([feature_of_reference_image, self.context_embed.weight[1].unsqueeze(
            0).repeat(feature_of_reference_image.shape[0], 1)], dim=0).unsqueeze(0) # 2x,c

        # feature_of_source_image_ = rearrange(feature_of_source_image.unsqueeze(0),
        #                                      "b c h w -> b (h w) c").contiguous() # 1 1024 256
        feature_of_source_image_ = feature_of_source_image.unsqueeze(0)
        feature_of_source_image_ = self.attn(
            feature_of_source_image_, feature_of_reference_image, )

        return feature_of_source_image_

    def context_maskpooling(self, feature, mask):
        '''
        get context feature tokens by maskpooling
        feature: [C, H/d, W/d]
        mask: [1, H, W]  [0,1]
        return: [token_num, C]
        '''
        mask[mask < 1] = 0
        mask = -1 * mask
        kernel_size = mask.shape[1] // feature.shape[1]
        mask = F.max_pool2d(mask, kernel_size=kernel_size,
                            stride=kernel_size, padding=0)

        index = mask == -1
        index = index.reshape(1, -1).squeeze()

        feature = feature.reshape(feature.shape[0], -1).permute(1, 0)
        feature = feature[index]

        return feature
# unused


def register_attention_control(model, controller, if_softmax=True):
    def ca_forward(self, place_in_unet):

        def forward(self, q, k, v):
            # Input projections
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

            # Separate into heads
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)

            # Attention
            _, _, _, c_per_head = q.shape
            # B x N_heads x N_tokens x N_tokens
            attn = q @ k.permute(0, 1, 3, 2)
            attn = attn / math.sqrt(c_per_head)

            if not if_softmax:
                controller(attn, True, place_in_unet)

            attn = torch.softmax(attn, dim=-1)

            if if_softmax:
                controller(attn, True, place_in_unet)

            # Get output
            out = attn @ v
            out = self._recombine_heads(out)
            out = self.out_proj(out)

            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    att_count = 0
    sub_nets = model.named_children()

    for net in sub_nets:

        cross_att_count += register_recr(net, 0, "up")

    controller.num_att_layers = att_count
