
from ptp.ptp_utils import text_under_image,view_images

from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
# from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from ptp.ptp_attention_controllers import EmptyControl,AttentionStore
from PIL import Image
from torch import tensor
from torch.nn import functional as F
def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                # cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def aggregate_attention_maskpooling(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, mask: tensor):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    
    mask[mask < 1] = 0
    mask = -1 * mask
    kernel_size = mask.shape[1] // res
    mask = F.max_pool2d(mask, kernel_size=kernel_size,
                        stride=kernel_size, padding=0)
    mask = -1*mask
    
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                # cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                
                cross_maps = cross_maps * mask
                
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    # for i in range(len(tokens)):
    for i in range(attention_maps.size(2)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        # image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0, mask=None):
    # show attention map on some res
    attention_maps = aggregate_attention_maskpooling(attention_store, res, from_where, False, select, mask).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1))