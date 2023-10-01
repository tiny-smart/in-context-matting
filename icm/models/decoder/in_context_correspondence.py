
import torch
import torch.nn as nn
from torch.nn import functional as F


def compute_correspondence_matrix(source_feature, ref_feature):
    """
    Compute correspondence matrix between source and reference features.
    Args:
        source_feature: [B, C, H, W]
        ref_feature: [B, C, H, W]
    Returns:
        correspondence_matrix: [B, H*W, H*W]
    """
    # [B, C, H, W] -> [B, H, W, C]
    source_feature = source_feature.permute(0, 2, 3, 1)
    ref_feature = ref_feature.permute(0, 2, 3, 1)

    # [B, H, W, C] -> [B, H*W, C]
    source_feature = torch.reshape(
        source_feature, [source_feature.shape[0], -1, source_feature.shape[-1]])
    ref_feature = torch.reshape(
        ref_feature, [ref_feature.shape[0], -1, ref_feature.shape[-1]])

    # norm
    source_feature = F.normalize(source_feature, p=2, dim=-1)
    ref_feature = F.normalize(ref_feature, p=2, dim=-1)

    # cosine similarity
    cos_sim = torch.matmul(
        source_feature, ref_feature.transpose(1, 2))  # [B, H*W, H*W]

    return cos_sim


def maskpooling(mask, res):
    '''
    Mask pooling to reduce the resolution of mask
    Input:
    mask: [B, 1, H, W]
    res: resolution
    Output: [B, 1, res, res]
    '''
    # mask[mask < 1] = 0
    mask[mask > 0] = 1
    mask = -1 * mask
    kernel_size = mask.shape[2] // res
    mask = F.max_pool2d(mask, kernel_size=kernel_size,
                        stride=kernel_size, padding=0)
    mask = -1*mask
    return mask


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out


def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    eroded = erode(mask, erode_kernel_size)
    dilated = dilate(mask, dilate_kernel_size)
    trimap = torch.zeros_like(mask)
    trimap[dilated == 1] = 0.5
    trimap[eroded == 1] = 1
    return trimap


def calculate_attention_score_(mask, attention_map, score_type):
    '''
    Calculate the attention score of the attention map
    mask: [B, H, W] value:0 or 1
    attention_map: [B, H*W, H, W]
    '''
    B, H, W = mask.shape
    mask_pos = mask.repeat(1, attention_map.shape[1], 1, 1)  # [B, H*W, H, W]
    score_pos = torch.sum(attention_map * mask_pos, dim=(2, 3))  # [B, H*W]
    score_pos = score_pos / torch.sum(mask_pos, dim=(2, 3))  # [B, H*W]

    mask_neg = 1 - mask_pos
    score_neg = torch.sum(attention_map * mask_neg, dim=(2, 3))
    score_neg = score_neg / torch.sum(mask_neg, dim=(2, 3))

    assert score_type in ['classification', 'softmax', 'ratio']

    if score_type == 'classification':
        score = torch.zeros_like(score_pos)
        score[score_pos > score_neg] = 1

    return score.reshape(B, H, W)


def refine_mask_by_attention(mask, attention_maps, iterations=10, score_type='classification'):
    '''
    mask: [B, H, W]
    attention_maps: [B, H*W, H, W]
    '''
    assert mask.shape[1] == attention_maps.shape[2]
    for i in range(iterations):
        score = calculate_attention_score_(
            mask, attention_maps, score_type=score_type)  # [B, H, W]

        if score.equal(mask):
            # print("iteration: ", i, "score equal to mask")
            break
        else:
            mask = score

    assert i != iterations - 1
    return mask


class InContextCorrespondence(nn.Module):
    '''
    one implementation of in_context_fusion

    forward(feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)
    '''

    def __init__(self,
                 use_bottle_neck=False,
                 in_dim=1280,
                 bottle_neck_dim=512,
                 refine_with_attention=False,
                 ):
        super().__init__()
        self.use_bottle_neck = use_bottle_neck
        self.refine_with_attention = refine_with_attention

    def forward(self, feature_of_reference_image, ft_attn_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W]
        ft_attn_of_source_image: {"ft": [B, C, H, W], "attn": [B, H_1, W_1, H_1*W_1]}
        guidance_on_reference_image: [B, 1, H_2, W_2]
        '''

        # get source_image h,w
        h, w = guidance_on_reference_image.shape[-2:]
        h_attn, w_attn = ft_attn_of_source_image['attn'].shape[-3:-1]

        feature_of_source_image = ft_attn_of_source_image['ft']
        attention_map_of_source_image = ft_attn_of_source_image['attn']

        cos_sim = compute_correspondence_matrix(
            feature_of_source_image, feature_of_reference_image)

        # 获得cos_sim每一行的最大值的索引
        index = torch.argmax(cos_sim, dim=-1)  # 1*1024

        mask_ref = maskpooling(guidance_on_reference_image,
                               h_attn)

        mask_ref = mask_ref.reshape(mask_ref.shape[0], -1)  # 1*1024

        new_index = torch.gather(mask_ref, 1, index)  # 1*1024
        res = int(new_index.shape[-1]**0.5)
        new_index = new_index.reshape(
            new_index.shape[0], res, res).unsqueeze(1)

        # resize mask_result to 512*512
        mask_result = new_index

        if self.refine_with_attention:
            mask_result = refine_mask_by_attention(
                mask_result, attention_map_of_source_image, iterations=10, score_type='classification')

        mask_result = F.interpolate(
            mask_result.float(), size=(h, w), mode='bilinear')

        # get trimap

        pesudo_trimap = generate_trimap(
            mask_result, erode_kernel_size=self.kernel_size, dilate_kernel_size=self.kernel_size)

        output = {}
        output['trimap'] = pesudo_trimap
        output['feature'] = feature_of_source_image
        output['mask'] = mask_result

        return output
