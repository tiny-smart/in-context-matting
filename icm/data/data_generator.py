import cv2
import os
import math
import numbers
import random
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms

from icm.util import instantiate_from_config
from icm.data.image_file import get_dir_ext
# one-hot or class, choice: [3, 1]
TRIMAP_CHANNEL = 1

RANDOM_INTERP = True

CUTMASK_PROB = 0

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
               cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if RANDOM_INTERP:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.phase = phase

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, trimap, mask = sample['image'][:, :, ::-
                                                     1], sample['alpha'], sample['trimap'], sample['mask']

        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1

        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        # normalize image
        image /= 255.

        if self.phase == "train":
            # convert GBR images to RGB
            fg = sample['fg'][:, :, ::-
                              1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:, :, ::-
                              1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
            # del sample['image_name']

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(
                alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        if TRIMAP_CHANNEL == 3:
            sample['trimap'] = F.one_hot(
                sample['trimap'], num_classes=3).permute(2, 0, 1).float()
        elif TRIMAP_CHANNEL == 1:
            sample['trimap'] = sample['trimap'][None, ...].float()
        else:
            raise NotImplementedError("TRIMAP_CHANNEL can only be 3 or 1")

        sample['mask'] = torch.from_numpy(mask).float()

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        "translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError(
                        "If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params(
                (0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(
                self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample

    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + \
            math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + \
            matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + \
            matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha == 0):
            return sample_ori
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(
            fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, mask, name = sample['fg'],  sample[
            'alpha'], sample['trimap'], sample['mask'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(
            bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0] / \
                h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)),
                                interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(
                    trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)),
                                interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                mask = cv2.resize(
                    mask, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(
            trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(
                0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0],
                     left_top[1]:left_top[1]+self.output_size[1], :]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0],
                           left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0],
                     left_top[1]:left_top[1]+self.output_size[1], :]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0],
                             left_top[1]:left_top[1]+self.output_size[1]]
        mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0],
                         left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap == 128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                              "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(
                fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(
                bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            mask_crop = cv2.resize(
                mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

        sample.update({'fg': fg_crop, 'alpha': alpha_crop,
                      'trimap': trimap_crop, 'mask': mask_crop, 'bg': bg_crop})
        return sample


class CropResize(object):
    # crop the image to square, and resize to target size
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, alpha, trimap, mask = sample['image'], sample['alpha'], sample['trimap'], sample['mask']
        # crop the image to square, and resize to target size

        h, w = img.shape[:2]
        if h == w:
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(
                mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        elif h > w:
            margin = (h-w)//2
            img = img[margin:margin+w, :]
            alpha = alpha[margin:margin+w, :]
            trimap = trimap[margin:margin+w, :]
            mask = mask[margin:margin+w, :]
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(
                mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        else:
            margin = (w-h)//2
            img = img[:, margin:margin+h]
            alpha = alpha[:, margin:margin+h]
            trimap = trimap[:, margin:margin+h]
            mask = mask[:, margin:margin+h]
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(
                mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        sample.update({'image': img_crop, 'alpha': alpha_crop,
                      'trimap': trimap_crop, 'mask': mask_crop})
        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(
            sample['image'], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_trimap = np.pad(
            sample['trimap'], ((0, pad_h), (0, pad_w)), mode="reflect")
        padded_mask = np.pad(
            sample['mask'], ((0, pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['mask'] = padded_mask

        return sample


class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        max_kernel_size = 30
        alpha = cv2.resize(alpha_ori, (640, 640),
                           interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        # generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(
            fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(
            bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        # generate mask
        low = 0.01
        high = 1.0
        thres = random.random() * (high - low) + low
        seg_mask = (alpha >= thres).astype(np.int).astype(np.uint8)
        random_num = random.randint(0, 3)
        if random_num == 0:
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 1:
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 2:
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 3:
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        seg_mask = cv2.resize(
            seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask

        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class CutMask(object):
    def __init__(self, perturb_prob=0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample['mask']  # H x W, trimap 0--255, segmask 0--1, alpha 0--1
        h, w = mask.shape
        perturb_size_h, perturb_size_w = random.randint(
            h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)

        mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1 +
                                                            perturb_size_h, y1:y1+perturb_size_w].copy()

        sample['mask'] = mask
        return sample


class DataGenerator(Dataset):
    def __init__(self, data, crop_size=512, phase="train"):
        self.phase = phase
        self.crop_size = crop_size
        self.alpha = data.alpha

        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = []
            self.trimap = []

        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        train_trans = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            CutMask(perturb_prob=CUTMASK_PROB),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train")]

        test_trans = [OriginScale(), ToTensor()]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def __getitem__(self, idx):
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])
            alpha = cv2.imread(
                self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)

            fg, alpha = self._composite_fg(fg, alpha, idx)

            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha,
                      'bg': bg, 'image_name': image_name}

        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            mask = (trimap >= 170).astype(np.float32)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                      'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape}

        sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(
                self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(
                fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(
                alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if np.any(alpha_tmp < 1):
                fg = fg.astype(
                    np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(
                fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(
                alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)


class MultiDataGeneratorDoubleSet(Dataset):
    # divide a dataset into train set and validation set
    def __init__(self, data, crop_size=1024, phase="train"):
        self.phase = phase
        self.crop_size = crop_size
        data = instantiate_from_config(data)

        if self.phase == "train":
            self.alpha = data.alpha_train
            self.merged = data.merged_train
            self.trimap = data.trimap_train

        elif self.phase == "val":
            self.alpha = data.alpha_val
            self.merged = data.merged_val
            self.trimap = data.trimap_val

        train_trans = [
            # RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),

            # CutMask(perturb_prob=CUTMASK_PROB),
            CropResize((self.crop_size, self.crop_size)),
            # RandomJitter(),
            ToTensor(phase="val")]

        # val_trans = [ OriginScale(), ToTensor() ]
        val_trans = [CropResize(
            (self.crop_size, self.crop_size)), ToTensor(phase="val")]

        self.transform = {
            'train':
                transforms.Compose(train_trans),

            'val':
                transforms.Compose(val_trans)
        }[phase]

        self.alpha_num = len(self.alpha)

    def __getitem__(self, idx):

        image = cv2.imread(self.merged[idx])
        alpha = cv2.imread(self.alpha[idx], 0)/255.
        trimap = cv2.imread(self.trimap[idx], 0)
        mask = (trimap >= 170).astype(np.float32)
        image_name = os.path.split(self.merged[idx])[-1]

        dataset_name = self.get_dataset_name(image_name)
        sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                  'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape, 'dataset_name': dataset_name}

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.alpha)

    def get_dataset_name(self, image_name):
        image_name = image_name.split('.')[0]
        if image_name.startswith('o_'):
            return 'AIM'
        elif image_name.endswith('_o') or image_name.endswith('_5k'):
            return 'PPM'
        elif image_name.startswith('m_'):
            return 'AM2k'
        elif image_name.endswith('_input'):
            return 'RWP636'
        elif image_name.startswith('p_'):
            return 'P3M'
        else:
            raise ValueError('image_name {} not recognized'.format(image_name))


class ContextDataset(Dataset):
    # divide a dataset into train set and validation set
    def __init__(self, data, crop_size=1024, phase="train"):
        self.phase = phase
        self.crop_size = crop_size
        data = instantiate_from_config(data)

        if self.phase == "train":
            self.dataset = data.dataset_train
            self.image_class_dict = data.image_class_dict_train

        elif self.phase == "val":
            self.dataset = data.dataset_val
            self.image_class_dict = data.image_class_dict_val

        # dict to list
        for key, value in self.image_class_dict.items():
            self.image_class_dict[key] = list(value.items())
        self.dataset = list(self.dataset.items())
        
        train_trans = [
            # RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),

            # CutMask(perturb_prob=CUTMASK_PROB),
            CropResize((self.crop_size, self.crop_size)),
            # RandomJitter(),
            ToTensor(phase="val")]

        # val_trans = [ OriginScale(), ToTensor() ]
        val_trans = [CropResize(
            (self.crop_size, self.crop_size)), ToTensor(phase="val")]

        self.transform = {
            'train':
                transforms.Compose(train_trans),

            'val':
                transforms.Compose(val_trans)
        }[phase]

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        
        image_name, image_info = self.dataset[idx]

        # get image sample
        dataset_name = image_info['dataset_name']
        image_sample = self.get_sample(image_name, dataset_name)

        # get context image
        class_name = str(
            image_info['class'])+'-'+str(image_info['sub_class'])+'-'+str(image_info['HalfOrFull'])
        (context_image_name, context_dataset_name) = self.image_class_dict[class_name][np.random.randint(
            len(self.image_class_dict[class_name]))]
        context_image_sample = self.get_sample(
            context_image_name, context_dataset_name)

        # merge image and context
        image_sample['context_image'] = context_image_sample['image']
        image_sample['context_guidance'] = context_image_sample['alpha']
        image_sample['context_image_name'] = context_image_sample['image_name']

        return image_sample

    def __len__(self):
        return len(self.dataset)

    def get_sample(self, image_name, dataset_name):
        cv2.setNumThreads(0)
        image_dir, label_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext = get_dir_ext(
            dataset_name)
        image_path = os.path.join(image_dir, image_name + merged_ext) if 'open-images' not in dataset_name else os.path.join(
            image_dir, image_name.split('_')[0] + merged_ext)
        label_path = os.path.join(label_dir, image_name + alpha_ext)
        trimap_path = os.path.join(trimap_dir, image_name + trimap_ext)

        image = cv2.imread(image_path)
        alpha = cv2.imread(label_path, 0)/255.
        trimap = cv2.imread(trimap_path, 0)
        mask = (trimap >= 170).astype(np.float32)
        image_name = os.path.split(image_path)[-1]

        sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                  'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape, 'dataset_name': dataset_name}

        sample = self.transform(sample)
        return sample
