from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from icm.criterion.matting_criterion_eval import compute_mse_loss_torch, compute_sad_loss_torch
from icm.util import instantiate_from_config
from pytorch_lightning.utilities import rank_zero_only
import os
import cv2
class InContextMatting(pl.LightningModule):
    '''
    In Context Matting Model
    consists of a feature extractor and a in-context decoder
    train model with lr, scheduler, loss
    '''

    def __init__(
        self,
        cfg_feature_extractor,
        cfg_in_context_decoder,
        cfg_loss_function,
        learning_rate,
        cfg_scheduler=None,
        **kwargs,
    ):
        super().__init__()

        self.feature_extractor = instantiate_from_config(
            cfg_feature_extractor)
        self.in_context_decoder = instantiate_from_config(
            cfg_in_context_decoder)

        self.loss_function = instantiate_from_config(cfg_loss_function)

        self.learning_rate = learning_rate
        self.cfg_scheduler = cfg_scheduler

    def forward(self, reference_images, guidance_on_reference_image, source_images):

        feature_of_reference_image = self.feature_extractor.get_reference_feature(
            reference_images)

        feature_of_source_image = self.feature_extractor.get_source_feature(
            source_images)
        
        reference = {'feature': feature_of_reference_image,
                     'guidance': guidance_on_reference_image}

        source = {'feature': feature_of_source_image, 'image': source_images}

        output, cross_map, self_map = self.in_context_decoder(source, reference)

        return output, cross_map, self_map

    def on_train_epoch_start(self):
        self.log("epoch", self.current_epoch, on_step=False,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss_dict, loss, _, _, _ = self.__shared_step(batch)

        self.__log_loss(loss_dict, loss, "train")

        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]
                 ["lr"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict, loss, preds, cross_map, self_map = self.__shared_step(batch)

        self.__log_loss(loss_dict, loss, "val")
        batch['cross_map'] = cross_map
        batch['self_map'] = self_map
        return preds, batch

    def __shared_step(self, batch):
        reference_images, guidance_on_reference_image, source_images, labels, trimaps = batch[
            "reference_image"], batch["guidance_on_reference_image"], batch["source_image"], batch["alpha"], batch["trimap"]

        outputs, cross_map, self_map = self(reference_images,
                       guidance_on_reference_image, source_images)
        
        sample_map = torch.zeros_like(trimaps)
        sample_map[trimaps==0.5] = 1     
        
        loss_dict = self.loss_function(sample_map, outputs, labels)

        loss = sum(loss_dict.values())
        if loss > 1e4 or torch.isnan(loss):
            raise ValueError(f"Loss explosion: {loss}")
        return loss_dict, loss, outputs, cross_map, self_map

    def __log_loss(self, loss_dict, loss, prefix):
        loss_dict = {
            f"{prefix}/{key}": loss_dict.get(key) for key in loss_dict}
        self.log_dict(loss_dict, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step_end(self, outputs):

        preds, batch = outputs
        h, w = batch['alpha_shape']
        
        
        cross_map = batch['cross_map']
        self_map = batch['self_map']
        # resize cross_map and self_map to the same size as preds
        cross_map = torch.nn.functional.interpolate(
            cross_map, size=preds.shape[2:], mode='bilinear', align_corners=False)
        self_map = torch.nn.functional.interpolate(
            self_map, size=preds.shape[2:], mode='bilinear', align_corners=False)
        
        # normalize cross_map and self_map
        cross_map = (cross_map - cross_map.min()) / \
            (cross_map.max() - cross_map.min())
        self_map = (self_map - self_map.min()) / \
            (self_map.max() - self_map.min())
        
        cross_map = cross_map[0].squeeze()*255.0
        self_map = self_map[0].squeeze()*255.0
        
        # get one sample from batch
        pred = preds[0].squeeze()*255.0
        source_image = batch['source_image'][0]
        label = batch["alpha"][0].squeeze()*255.0
        trimap = batch["trimap"][0].squeeze()*255.0
        trimap[trimap == 127.5] = 128
        reference_image = batch["reference_image"][0]
        guidance_on_reference_image = batch["guidance_on_reference_image"][0]
        dataset_name = batch["dataset_name"][0]
        image_name = batch["image_name"][0].split('.')[0]

        # save pre to model.val_save_path
        
        # if self.val_save_path is not None:
        if hasattr(self, 'val_save_path'):
            os.makedirs(self.val_save_path, exist_ok=True)
            # resize preds to h,w
            pred_ = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
            pred_ = pred_.squeeze().cpu().numpy()
            pred_ = pred_.astype('uint8')
            cv2.imwrite(os.path.join(self.val_save_path, image_name+'.png'), pred_)
            
        masked_reference_image = reference_image*guidance_on_reference_image
        # self.__compute_and_log_mse_sad_of_one_sample(
        #     pred, label, trimap, prefix="val")

        self.__log_image(
            source_image, masked_reference_image, pred, label, dataset_name, image_name, prefix='val', self_map=self_map, cross_map=cross_map)


    # def validation_step_end(self, outputs):

    #     preds, batch = outputs

    #     cross_map = batch['cross_map']
    #     self_map = batch['self_map']
    #     # resize cross_map and self_map to the same size as preds
    #     cross_map = torch.nn.functional.interpolate(
    #         cross_map, size=preds.shape[2:], mode='bilinear', align_corners=False)
    #     self_map = torch.nn.functional.interpolate(
    #         self_map, size=preds.shape[2:], mode='bilinear', align_corners=False)
        
    #     # normalize cross_map and self_map
    #     cross_map = (cross_map - cross_map.min()) / \
    #         (cross_map.max() - cross_map.min())
    #     self_map = (self_map - self_map.min()) / \
    #         (self_map.max() - self_map.min())
        
    #     cross_map = cross_map[0].squeeze()*255.0
    #     self_map = self_map[0].squeeze()*255.0
        
    #     # get one sample from batch
    #     pred = preds[0].squeeze()*255.0
    #     source_image = batch['source_image'][0]
    #     label = batch["alpha"][0].squeeze()*255.0
    #     trimap = batch["trimap"][0].squeeze()*255.0
    #     trimap[trimap == 127.5] = 128
    #     reference_image = batch["reference_image"][0]
    #     guidance_on_reference_image = batch["guidance_on_reference_image"][0]
    #     dataset_name = batch["dataset_name"][0]
    #     image_name = batch["image_name"][0].split('.')[0]

    #     masked_reference_image = reference_image*guidance_on_reference_image

    #     self.__compute_and_log_mse_sad_of_one_sample(
    #         pred, label, trimap, prefix="val")

    #     self.__log_image(
    #         source_image, masked_reference_image, pred, label, dataset_name, image_name, prefix='val', self_map=self_map, cross_map=cross_map)

    def __compute_and_log_mse_sad_of_one_sample(self, pred, label, trimap, prefix="val"):
        # compute loss for unknown pixels
        mse_loss_unknown_ = compute_mse_loss_torch(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss_torch(pred, label, trimap)

        # compute loss for all pixels
        trimap = torch.ones_like(label)*128
        mse_loss_all_ = compute_mse_loss_torch(pred, label, trimap)
        sad_loss_all_ = compute_sad_loss_torch(pred, label, trimap)

        # log
        metrics_unknown = {f'{prefix}/mse_unknown': mse_loss_unknown_,
                           f'{prefix}/sad_unknown': sad_loss_unknown_, }

        metrics_all = {f'{prefix}/mse_all': mse_loss_all_,
                       f'{prefix}/sad_all': sad_loss_all_, }

        self.log_dict(metrics_unknown, on_step=False,
                      on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics_all, on_step=False,
                      on_epoch=True, prog_bar=False, sync_dist=True)

    def __log_image(self, source_image, masked_reference_image, pred, label, dataset_name, image_name, prefix='val', self_map=None, cross_map=None):
        ########### log source_image, masked_reference_image, output and gt ###########
        # process image, masked_reference_image, pred, label
        source_image = self.__revert_normalize(source_image)
        masked_reference_image = self.__revert_normalize(
            masked_reference_image)
        pred = torch.stack((pred/255.0,)*3, axis=-1)
        label = torch.stack((label/255.0,)*3, axis=-1)
        self_map = torch.stack((self_map/255.0,)*3, axis=-1)
        cross_map = torch.stack((cross_map/255.0,)*3, axis=-1)
        
        # concat pred, masked_reference_image, label, source_image
        image_for_log = torch.stack(
            (source_image, masked_reference_image, label, pred, self_map, cross_map), axis=0)

        # log image
        self.logger.experiment.add_images(
            f'{prefix}-{dataset_name}/{image_name}', image_for_log, self.current_epoch, dataformats='NHWC')

    def __revert_normalize(self, image):
        # image: [C, H, W]
        image = image.permute(1, 2, 0)
        image = image * torch.tensor([0.229, 0.224, 0.225], device=self.device) + \
            torch.tensor([0.485, 0.456, 0.406], device=self.device)
        image = torch.clamp(image, 0, 1)
        return image

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        torch.cuda.empty_cache()
    def test_step(self, batch, batch_idx):
        loss_dict, loss, preds = self.__shared_step(batch)

        return loss_dict, loss, preds

    def configure_optimizers(self):
        params = self.__get_trainable_params()
        opt = torch.optim.Adam(params, lr=self.learning_rate)

        if self.cfg_scheduler is not None:
            scheduler = self.__get_scheduler(opt)
            return [opt], scheduler
        return opt

    def __get_trainable_params(self):
        params = []
        params = params + self.in_context_decoder.get_trainable_params() + \
            self.feature_extractor.get_trainable_params()
        return params

    def __get_scheduler(self, opt):
        scheduler = instantiate_from_config(self.cfg_scheduler)
        scheduler = [
            {
                "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return scheduler

from pytorch_lightning.callbacks import ModelCheckpoint

class ModifiedModelCheckpoint(ModelCheckpoint):
    def delete_frozen_params(self, ckpt):
        # delete params with requires_grad=False
        for k in list(ckpt["state_dict"].keys()):
            # remove ckpt['state_dict'][k] if 'feature_extractor' in k
            if "feature_extractor" in k:
                del ckpt["state_dict"][k]
        return ckpt

    def _save_model(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_model(trainer, filepath)

        if trainer.is_global_zero:
            ckpt = torch.load(filepath)
            ckpt = self.delete_frozen_params(ckpt)
            torch.save(ckpt, filepath)