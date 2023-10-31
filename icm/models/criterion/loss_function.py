import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import focal_loss

class LossFunction(nn.Module):
    '''
    Loss function set
    losses=['unknown_l1_loss', 'known_l1_loss',
            'loss_pha_laplacian', 'loss_gradient_penalty',
            'smooth_l1_loss', 'cross_entropy_loss', 'focal_loss']
    '''
    def __init__(self,
                 *,
                 losses,
                 ):
        super(LossFunction, self).__init__()
        self.losses = losses

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        # scale = 1

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def smooth_l1_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.smooth_l1_loss(preds['phas'], targets['phas'])

        return dict(smooth_l1_loss=loss)

    def known_smooth_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.smooth_l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)
    
    def cross_entropy_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.binary_cross_entropy_with_logits(preds['phas'], targets['phas'])

        return dict(cross_entropy_loss=loss)
    
    def focal_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = focal_loss.sigmoid_focal_loss(preds['phas'], targets['phas'], reduction='mean')

        return dict(focal_loss=loss)
    def forward(self, sample_map, preds, targets):
        
        preds = {'phas': preds}
        targets = {'phas': targets}
        losses = dict()
        for k in self.losses:
            if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k=='known_smooth_l1_loss':
                losses.update(getattr(self, k)(sample_map, preds, targets))
            else:
                losses.update(getattr(self, k)(preds, targets))
        return losses

class LossFunction2(nn.Module):
    '''
    Loss function set
    losses=['unknown_l1_loss', 'known_l1_loss',
            'loss_pha_laplacian', 'loss_gradient_penalty',
            'smooth_l1_loss', 'cross_entropy_loss', 'focal_loss']
    '''
    def __init__(self,
                 *,
                 losses_seg = ['known_smooth_l1_loss'],
                 losses_matting = ['unknown_l1_loss', 'known_l1_loss','loss_pha_laplacian', 'loss_gradient_penalty',],
                 ):
        super(LossFunction2, self).__init__()
        self.losses_seg = losses_seg
        self.losses_matting = losses_matting

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        # scale = 1

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def smooth_l1_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.smooth_l1_loss(preds['phas'], targets['phas'])

        return dict(smooth_l1_loss=loss)

    def known_smooth_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.smooth_l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)
    
    def cross_entropy_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.binary_cross_entropy_with_logits(preds['phas'], targets['phas'])

        return dict(cross_entropy_loss=loss)
    
    def focal_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = focal_loss.sigmoid_focal_loss(preds['phas'], targets['phas'], reduction='mean')

        return dict(focal_loss=loss)
    def forward_single_sample(self, sample_map, preds, targets):
        # check if targets only have element 0 and 1
        if torch.all(targets == 0) or torch.all(targets == 1):
            
            preds = {'phas': preds}
            targets = {'phas': targets}
            losses = dict()
            for k in self.losses_seg:
                if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k=='known_smooth_l1_loss':
                    losses.update(getattr(self, k)(sample_map, preds, targets))
                else:
                    losses.update(getattr(self, k)(preds, targets))
            return losses
        else:
            preds = {'phas': preds}
            targets = {'phas': targets}
            losses = dict()
            for k in self.losses_matting:
                if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k=='known_smooth_l1_loss':
                    losses.update(getattr(self, k)(sample_map, preds, targets))
                else:
                    losses.update(getattr(self, k)(preds, targets))
            return losses
        
    def forward(self, sample_map, preds, targets):
        losses = dict()
        for i in range(preds.shape[0]):
            losses_ = self.forward_single_sample(sample_map[i].unsqueeze(0), preds[i].unsqueeze(0), targets[i].unsqueeze(0))
            for k in losses_:
                if k in losses:
                    losses[k] += losses_[k]
                else:
                    losses[k] = losses_[k]
        return losses
#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]