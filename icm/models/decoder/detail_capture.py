import torch
from torch import nn
from torch.nn import functional as F

class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Basic_Conv3x3_attn(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        res = False,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.ln = nn.LayerNorm(in_chans, elementwise_affine=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(x)
        x = self.conv(x)

        return x
    
# class Basic_Conv3x3_attn(nn.Module):
#     """
#     Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
#     """
#     def __init__(
#         self,
#         in_chans,
#         out_chans,
#         res = False,
#         stride=1,
#         padding=1,
#     ):
#         super().__init__()
#         self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
#         self.ln = nn.LayerNorm([in_chans, res, res], elementwise_affine=True)
#         self.relu = nn.ReLU(True)

#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)
#         x = self.ln(x)
#         x = self.relu(x)
#         x = self.conv(x)

#         return x
    
class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans
        self.conv_chans.insert(0, in_chans)
        
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_, stride=2)
            )
    
    def forward(self, x):
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            check = self.convs[i]
            name_ = 'D'+str(i+1)
            out_dict[name_] = x
        
        return out_dict

class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out    

class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(
        self,
        in_chans = 32,
        mid_chans = 16,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
            )

    def forward(self, x):
        x = self.matting_convs(x)

        return x

# TODO: implement groupnorm and ws. In ODISE, bs=2, they work well; when bs = 1, mse loss will be nan, why?
class DetailCapture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        in_chans = 384,
        img_chans=4,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
        ckpt=None,
        use_sigmoid = True,
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans = img_chans)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                )
            )

        self.matting_head = Matting_Head(
            in_chans = fusion_out[-1],
        )
        
        if ckpt != None and ckpt != '':
            self.load_state_dict(ckpt['state_dict'], strict=False)
            print('load detail capture ckpt from', ckpt['path'])
        
        self.use_sigmoid = use_sigmoid
        self.img_chans = img_chans
    def forward(self, features, images):
        
        if isinstance(features, dict):
            
            trimap = features['trimap']
            features = features['feature']
            if self.img_chans ==4:
                images = torch.cat([images, trimap], dim=1)
        
        detail_features = self.convstream(images)
        # D0  2  4  512 512
        # D1  2  48 256 256
        # D2  2  96 128 128
        # D3  2  192 64 64
        for i in range(len(self.fusion_blks)):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = self.fusion_blks[i](features, detail_features[d_name_])
        
        if self.use_sigmoid:
            phas = torch.sigmoid(self.matting_head(features))
        else:
            phas = self.matting_head(features)
        return phas
    
    def get_trainable_params(self):
        return list(self.parameters())
    
class MaskDecoder(nn.Module):
    '''
    use trans-conv to decode mask
    '''
    def __init__(
        self,
        in_chans = 384,
    ):
        super().__init__()
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(in_chans, in_chans // 4, kernel_size=2, stride=2),
            # LayerNorm2d(in_chans // 4),
            nn.BatchNorm2d(in_chans // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_chans // 4, in_chans // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_chans // 8),
            nn.ReLU(),       
        )
        
        self.matting_head = Matting_Head(
            in_chans = in_chans // 8,
        )
        
    def forward(self, x, images):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.output_upscaling(x)
        x = self.matting_head(x)
        x = torch.sigmoid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x