from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.optimizer import Optimizer
from icm.models.criterion.matting_criterion import MattingCriterion
from icm.models.criterion.matting_criterion_eval import compute_mse_loss, compute_sad_loss, compute_connectivity_error, compute_gradient_loss, compute_mse_loss_torch, compute_sad_loss_torch
from icm.util import instantiate_from_config, instantiate_feature_extractor
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import grad_norm
from torch.nn import functional as F
from torchvision.ops import focal_loss


class InContextMatting(pl.LightningModule):
    def __init__(
        self,
        cfg_feature_extractor,
        cfg_decoder,
        feature_index,
        learning_rate,
        use_scheduler,
        scheduler_config,
        train_adapter_params,
        freeze_transformer=False,
        context_type='maskpooling',  # unused, move to cfg_decoder
        loss_type='vit_matte',  # 'vit_matte' or 'smooth_l1'
    ):
        super().__init__()

        # init model, move to configure_shared_model, move back to init
        self.feature_extractor = instantiate_feature_extractor(
            cfg_feature_extractor)

        self.in_context_decoder = instantiate_from_config(cfg_decoder)

        self.feature_index = feature_index
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.scheduler_config = scheduler_config
        self.train_adapter_params = train_adapter_params
        # self.context_type = context_type
        self.criterion = MattingCriterion(
            losses=['unknown_l1_loss', 'known_l1_loss',
                    'loss_pha_laplacian', 'loss_gradient_penalty']
        )
        self.loss_type = loss_type
        self.freeze_transformer = freeze_transformer
        # if self.context_type == 'embed':
        #     self.context_embed = nn.Embedding(
        #         2, cfg_decoder["params"]['in_chans'])

    def on_train_start(self):
        # set layers to get features
        self.feature_extractor.reset_dim_stride()

    def on_train_epoch_start(self, unused=None):
        self.log("epoch", self.current_epoch, on_step=False,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, images, context):
        x = self.feature_extractor({'img': images})[self.feature_index]
        x = self.in_context_decoder(x, context, images)

        return x

    def shared_step(self, batch, batch_idx):
        context_images, context_masks, images = batch[
            "context_image"], batch["context_guidance"], batch["image"]

        context_feature = self.feature_extractor({'img': context_images})[
            self.feature_index].detach()

        # if self.context_type == 'maskpooling':
        #     context_feature = self.context_maskpooling(
        #         context_feature, context_masks)
        # elif self.context_type == 'embed':
        #     # resize context_masks to [B, 1, H/d, W/d]
        #     context_masks = F.interpolate(
        #         context_masks, size=context_feature.shape[2:], mode='nearest')
        #     # add self.context_embed[0] to pixels where context_masks == 0, add self.context_embed[1] to pixels where context_masks == 1
        #     # embedding = self.context_embed(context_masks.squeeze(1).long()).permute(0, 3, 1, 2)
        #     context_feature = context_feature + \
        #         self.context_embed(context_masks.squeeze(
        #             1).long()).permute(0, 3, 1, 2)
        #     # flatten context_feature
        #     context_feature = context_feature.reshape(
        #         context_feature.shape[0], context_feature.shape[1], -1).permute(0, 2, 1)
        # resize context_masks to [B, 1, H/d, W/d]
        context_masks = F.interpolate(
            context_masks, size=context_feature.shape[2:], mode='nearest')
        context = {'feature': context_feature, 'mask': context_masks}
        output = self(images, context)

        return output

    # unused, move to context_decoder
    def context_maskpooling(self, feature, mask):
        '''
        get context feature tokens by maskpooling
        feature: [B, C, H/d, W/d]
        mask: [B, 1, H, W]  [0,1]
        return: [B, token_num, C] token_num = H*W/d^2
        '''
        mask[mask < 1] = 0
        mask = -1 * mask
        kernel_size = mask.shape[2] // feature.shape[2]
        mask = F.max_pool2d(mask, kernel_size=kernel_size,
                            stride=kernel_size, padding=0)
        mask = -1*mask

        feature = mask*feature
        feature = feature.reshape(
            feature.shape[0], feature.shape[1], -1).permute(0, 2, 1)

        return feature

    def training_step(self, batch, batch_idx):
        labels, trimaps = batch["alpha"], batch["trimap"]
        output = self.shared_step(batch, batch_idx)

        sample_map = torch.zeros_like(trimaps)
        sample_map[trimaps == 1] = 1
        if self.loss_type == 'vit_matte':
            losses = self.criterion(
                sample_map, {"phas": output}, {"phas": labels})
        elif self.loss_type == 'smooth_l1':
            losses = F.smooth_l1_loss(output, labels, reduction='none')
            losses = {'smooth_l1_loss': losses.mean()}
        elif self.loss_type == 'cross_entropy':
            losses = F.binary_cross_entropy_with_logits(output, labels)
            losses = {'cross_entropy': losses.mean()}
        elif self.loss_type == 'focal_loss':
            losses = focal_loss.sigmoid_focal_loss(output, labels)
            losses = {'focal_loss': losses.mean()}
        # log training loss

        # add prefix 'train' to the keys
        losses = {f"train/{key}": losses.get(key) for key in losses}

        self.log_dict(losses, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=True)

        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]
                 ["lr"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # init loss tensor
        loss = torch.zeros(1).type_as(labels)

        for key in losses:
            loss += losses[key]

        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        labels, trimaps, dataset_name, image_name, context_image, context_guidance = batch["alpha"], batch[
            "trimap"], batch["dataset_name"], batch["image_name"], batch["context_image"], batch["context_guidance"]

        output = self.shared_step(batch, batch_idx)

        sample_map = torch.zeros_like(trimaps)
        sample_map[trimaps == 1] = 1
        if self.loss_type == 'vit_matte':
            losses = self.criterion(
                sample_map, {"phas": output}, {"phas": labels})
        elif self.loss_type == 'smooth_l1':
            losses = F.smooth_l1_loss(output, labels, reduction='none')
            losses = {'smooth_l1_loss': losses.mean()}
        elif self.loss_type == 'cross_entropy':
            losses = F.binary_cross_entropy_with_logits(output, labels)
            losses = {'cross_entropy': losses.mean()}
        elif self.loss_type == 'focal_loss':
            losses = focal_loss.sigmoid_focal_loss(output, labels)
            losses = {'focal_loss': losses.mean()}
        # log training loss

        # add prefix 'train' to the keys
        losses = {f"val/{key}": losses.get(key) for key in losses}

        self.log_dict(losses, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=True)

        # init loss tensor
        loss = torch.zeros(1).type_as(labels)

        for key in losses:
            loss += losses[key]

        self.log("val/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return output, batch

    def validation_step_end(self, outputs):

        output, batch = outputs
        labels, trimaps, dataset_name, image_name, context_image, context_guidance = batch["alpha"][0].unsqueeze(0), batch[
            "trimap"][0].unsqueeze(0), batch["dataset_name"], batch["image_name"], batch["context_image"][0].unsqueeze(0), batch["context_guidance"][0].unsqueeze(0)

        label = labels.squeeze()*255.0
        trimap = trimaps.squeeze()*128
        pred = output[0].squeeze()*255.0

        # for logging
        guidance_image = context_image*context_guidance
        guidance_image = guidance_image.squeeze()
        dataset_name = dataset_name[0]
        image_name = image_name[0].split('.')[0]
        image = batch['image'][0]

        # compute loss

        metrics_unknown, metrics_all = self.compute_two_metrics(
            pred, label, trimap)

        # log validation metrics
        self.log_dict(metrics_unknown, on_step=False,
                      on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics_all, on_step=False,
                      on_epoch=True, prog_bar=False, sync_dist=True)

        self.log_validation_result(
            image, guidance_image, pred, label, dataset_name, image_name)

    def compute_two_metrics(self, pred, label, trimap, prefix="val"):
        # compute loss for unknown pixels
        mse_loss_unknown_ = compute_mse_loss_torch(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss_torch(
            pred, label, trimap)[0]

        # compute loss for all pixels
        trimap = torch.ones_like(label)*128

        mse_loss_all_ = compute_mse_loss_torch(pred, label, trimap)
        sad_loss_all_ = compute_sad_loss_torch(
            pred, label, trimap)[0]

        # log validation metrics
        metrics_unknown = {f'{prefix}/mse_unknown': mse_loss_unknown_,
                           f'{prefix}/sad_unknown': sad_loss_unknown_, }

        metrics_all = {f'{prefix}/mse_all': mse_loss_all_,
                       f'{prefix}/sad_all': sad_loss_all_, }

        return metrics_unknown, metrics_all

    def log_validation_result(self, image, guidance_image, pred, label, dataset_name, image_name):
        ########### log image, guidance_image, output and gt ###########
        # process image
        image = image.permute(1, 2, 0)
        image = image * torch.tensor([0.229, 0.224, 0.225], device=self.device) + \
            torch.tensor([0.485, 0.456, 0.406], device=self.device)
        # clip to [0, 1]
        image = torch.clamp(image, 0, 1)

        # process guidance_image, pred, label
        guidance_image = guidance_image.permute(1, 2, 0)
        guidance_image = guidance_image * \
            torch.tensor([0.229, 0.224, 0.225], device=self.device) + \
            torch.tensor([0.485, 0.456, 0.406], device=self.device)
        guidance_image = torch.clamp(guidance_image, 0, 1)

        pred = torch.stack((pred/255.0,)*3, axis=-1)
        label = torch.stack((label/255.0,)*3, axis=-1)

        # concat pred, guidance_image, label, image
        image_to_log = torch.stack(
            (image, guidance_image, label, pred), axis=0)

        # log image
        self.logger.experiment.add_images(
            f'validation-{dataset_name}/{image_name}', image_to_log, self.current_epoch, dataformats='NHWC')

    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.in_context_decoder.parameters() if not self.freeze_transformer else self.in_context_decoder.freeze_transformer()

        if self.train_adapter_params:
            params = list(params)
            adapter_params = self.feature_extractor.get_trainable_params()
            params = params + adapter_params
        opt = torch.optim.Adam(params, lr=lr)

        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt
