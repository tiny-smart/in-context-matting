
from torch import einsum
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


class TrainingFreeAttention:
    def __init__(self, res_ratio=4, pool_type='average', temp_softmax=1, use_scale=False, upsample_mode='bilinear') -> None:
        self.res_ratio = res_ratio
        self.pool_type = pool_type
        self.temp_softmax = temp_softmax
        self.use_scale = use_scale
        self.upsample_mode = upsample_mode

    def __call__(self, features, features_ref, roi_mask,):
        # roi_mask: [B, 1, H, W]
        # features: [B, C, h, w]
        # features_ref: [B, C, h, w]
        B, _, H, W = roi_mask.shape
        H_attn, W_attn = H//self.res_ratio, W//self.res_ratio
        features, features_ref = self.resize_input_to_res(
            features, features_ref, (H, W))  # [H//res_ratio, W//res_ratio]

        # List, len = B, each element: [C_q, dim], dim = H//res_ratio * W//res_ratio
        features_ref = self.get_roi_features(features_ref, roi_mask)

        features = features.reshape(
            B, -1, features.shape[2] * features.shape[3]).permute(0, 2, 1)  # [B, C, dim]
        # List, len = B, each element: [C_q, C]
        attn_output = self.compute_attention(features, features_ref)

        # List, len = B, each element: [C_q, H_attn, W_attn]
        attn_output = self.reshape_attn_output(attn_output, (H_attn, W_attn))

        return attn_output

    def resize_input_to_res(self, features, features_ref, size):
        # features: [B, C, h, w]
        # features_ref: [B, C, h, w]
        H, W = size
        target_H, target_W = H//self.res_ratio, W//self.res_ratio
        features = F.interpolate(features, size=(
            target_H, target_W), mode=self.upsample_mode)
        features_ref = F.interpolate(features_ref, size=(
            target_H, target_W), mode=self.upsample_mode)
        return features, features_ref

    def get_roi_features(self, feature, mask):
        '''
        get feature tokens by maskpool
        feature: [B, C, h, w]
        mask: [B, 1, H, W]  [0,1]
        return: List, len = B, each element: [token_num, C]
        '''

        # assert mask only has elements 0 and 1
        assert torch.all(torch.logical_or(mask == 0, mask == 1))
        assert mask.max() == 1 and mask.min() == 0

        B, _, H, W = mask.shape
        h, w = feature.shape[2:]

        output = []
        for i in range(B):
            mask_ = mask[i]
            feature_ = feature[i]
            feature_ = self.maskpool(feature_, mask_)
            output.append(feature_)
        return output

    def maskpool(self, feature, mask):
        '''
        get feature tokens by maskpool
        feature: [C, h, w]
        mask: [1, H, W]  [0,1]
        return: [token_num, C]
        '''

        if self.pool_type == 'max':
            mask = F.max_pool2d(mask, kernel_size=self.res_ratio,
                                stride=self.res_ratio, padding=0)
        elif self.pool_type == 'average':
            mask = F.avg_pool2d(mask, kernel_size=self.res_ratio,
                                stride=self.res_ratio, padding=0)
        elif self.pool_type == 'min':
            mask = -1*mask
            mask = F.max_pool2d(mask, kernel_size=self.res_ratio,
                                stride=self.res_ratio, padding=0)
            mask = -1*mask
        else:
            raise NotImplementedError

        # element-wise multiplication mask and feature
        feature = feature * mask

        index = (mask > 0).reshape(1, -1).squeeze()
        feature = feature.reshape(feature.shape[0], -1).permute(1, 0)

        feature = feature[index]
        return feature

    def compute_attention(self, features, features_ref):
        '''
        features: [B, C, dim]
        features_ref: List, len = B, each element: [C_q, dim]
        return: List, len = B, each element: [C_q, C]
        '''
        output = []
        for i in range(len(features_ref)):
            feature_ref = features_ref[i]
            feature = features[i]
            feature = self.compute_attention_single(feature, feature_ref)
            output.append(feature)
        return output

    def compute_attention_single(self, feature, feature_ref):
        '''
        compute attention with softmax
        feature: [C, dim]
        feature_ref: [C_q, dim]
        return: [C_q, C]
        '''
        scale = feature.shape[-1]**-0.5 if self.use_scale else 1.0
        sim = einsum('i d, j d -> i j', feature_ref, feature)*scale
        sim = sim/self.temp_softmax
        sim = sim.softmax(dim=-1)
        return sim

    def reshape_attn_output(self, attn_output, attn_size):
        '''
        attn_output: List, len = B, each element: [C_q, C]
        return: List, len = B, each element: [C_q, H_attn, W_attn]
        '''
        # attn_output[0].shape[1] sqrt to get H_attn, W_attn
        H_attn, W_attn = attn_size

        output = []
        for i in range(len(attn_output)):
            attn_output_ = attn_output[i]
            attn_output_ = attn_output_.reshape(
                attn_output_.shape[0], H_attn, W_attn)
            output.append(attn_output_)
        return output


class TrainingFreeAttentionBlocks(nn.Module):
    '''
    one implementation of in_context_fusion

    forward(feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)
    '''

    def __init__(self,
                 res_ratio=8,
                 pool_type='min',
                 temp_softmax=1000,
                 use_scale=False,
                 upsample_mode='bicubic',
                 bottle_neck_dim=None,

                 ):
        super().__init__()

        self.attn_module = TrainingFreeAttention(res_ratio=res_ratio,
                                                 pool_type=pool_type,
                                                 temp_softmax=temp_softmax,
                                                 use_scale=use_scale,
                                                 upsample_mode=upsample_mode)

    def forward(self, feature_of_reference_image, ft_attn_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W]
        ft_attn_of_source_image: {"ft_cor": [B, C, H, W], "attn": [B, H_1, W_1, H_1*W_1], "ft_matting": [B, C, H, W]}
        guidance_on_reference_image: [B, 1, H_2, W_2]
        '''
        assert feature_of_reference_image.shape[0] == 1 
        # get source_image h,w
        h, w = guidance_on_reference_image.shape[-2:]

        features_cor = ft_attn_of_source_image['ft_cor']
        features_matting = ft_attn_of_source_image['ft_matting']
        features_ref = feature_of_reference_image

        guidance_on_reference_image[guidance_on_reference_image > 0.5] = 1
        guidance_on_reference_image[guidance_on_reference_image <= 0.5] = 0
        attn_output = self.attn_module(
            features_cor, features_ref, guidance_on_reference_image)
        
        
        attn_output = attn_output[0].sum(dim=0).unsqueeze(0).unsqueeze(0)

        self_attn_output = self.training_free_self_attention(
            attn_output, ft_attn_of_source_image['attn'])

        # resize
        self_attn_output = F.interpolate(
            self_attn_output, size=(h, w), mode='bilinear')

        output = {}
        output['trimap'] = self_attn_output
        output['feature'] = features_matting
        output['mask'] = attn_output

        return output

    def training_free_self_attention(self, x, self_attn_maps):
        '''
        Compute self-attention using the attention maps.

        Parameters:
        x (torch.Tensor): The input tensor. Shape: [B, 1, H, W]
        self_attn_maps (torch.Tensor): The attention maps. Shape: [B, H1, W1, H1*W1]

        Returns:
        torch.Tensor: The result of the self-attention computation.
        '''

        # Original dimensions of x
        # Assuming x's shape is [B, 1, H, W] based on your comment
        B, _, H, W = x.shape

        # Dimensions of the attention maps
        _, H1, W1, _ = self_attn_maps.shape

        # Resize x to match the spatial dimensions of the attention maps
        # You might need align_corners depending on your version of PyTorch
        x = F.interpolate(x, size=(H1, W1), mode='bilinear',
                          align_corners=True)

        # Reshape the attention maps and x for matrix multiplication
        # Reshaping from [B, H1, W1, H1*W1] to [B, H1*W1, H1*W1]
        self_attn_maps = self_attn_maps.view(B, H1 * W1, H1 * W1)
        # Reshaping from [B, 1, H1, W1] to [B, 1, H1*W1]
        x = x.view(B, 1, H1 * W1)

        # Apply the self-attention mechanism
        # Matrix multiplication between the attention maps and the input feature map
        # This step essentially computes the weighted sum of feature vectors in the input,
        # where the weights are defined by the attention maps.
        # Multiplying with the transpose to get shape [B, 1, H1*W1]
        out = torch.matmul(x, self_attn_maps.transpose(1, 2))

        # Reshape the output tensor to the original spatial dimensions
        out = out.view(B, 1, H1, W1)  # Reshaping back to spatial dimensions

        # # Resize the output back to the input's original dimensions (if necessary)
        # out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        return out
