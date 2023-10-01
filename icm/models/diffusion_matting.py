
from icm.models.criterion.matting_criterion_eval import compute_mse_loss_torch, compute_sad_loss_torch
from icm.util import instantiate_from_config, instantiate_feature_extractor
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR


class DiffusionMatting(pl.LightningModule):
    def __init__(
        self,
        cfg_feature_extractor,
        cfg_decoder,
        cfg_loss_function,
        learning_rate,
        guidance_type,
        cfg_scheduler,
    ):
        super().__init__()

        # init model, move to configure_shared_model, move back to init
        self.feature_extractor = instantiate_feature_extractor(
            cfg_feature_extractor)

        self.decoder = instantiate_from_config(cfg_decoder)

        self.guidance_type = guidance_type
        self.learning_rate = learning_rate

        self.scheduler_config = cfg_scheduler

        self.loss_function = instantiate_from_config(cfg_loss_function)

    def forward(self, images, images_guidance):
        x = self.feature_extractor.get_source_feature(images)
        x = self.diffusion_decoder(x, images_guidance)
        return x
    
    def on_train_start(self):
        # set layers to get features
        self.feature_extractor.reset_dim_stride()

    def on_train_epoch_start(self, unused=None):
        self.log("epoch", self.current_epoch, on_step=False,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss_dict, loss, _ = self.__shared_step(batch)

        self.__log_loss(loss_dict, loss, "train")

        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]
                 ["lr"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def __log_loss(self, loss_dict, loss, prefix):
        loss_dict = {
            f"{prefix}/{key}": loss_dict.get(key) for key in loss_dict}
        self.log_dict(loss_dict, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def __get_guidance_map(self, trimaps, labels):
        if self.guidance_type == "coarse_map":
            # make coarse mask, 0 for bg, 1 for fg and unknown
            coarse_mask = torch.zeros_like(trimaps)
            coarse_mask[trimaps == 1] = 1
            coarse_mask[labels > 0.5] = 1
            return coarse_mask

        elif self.guidance_type == "trimap":
            return trimaps

        else:
            raise NotImplementedError

    def __shared_step(self, batch, batch_idx):
        images, labels, trimaps = batch["image"], batch["alpha"], batch["trimap"]

        guidance_map = self.__get_guidance_map(trimaps, labels)
        images_guidance = torch.cat((images, guidance_map), dim=1)

        outputs = self(images, images_guidance)
        loss_dict = self.loss_function(trimaps, outputs, labels)

        loss = sum(loss_dict.values())

        return loss_dict, loss, outputs
    
    def validation_step(self, batch, batch_idx):
        loss_dict, loss, preds = self.__shared_step(batch)

        self.__log_loss(loss_dict, loss, "val")

        return preds, batch

    def validation_step_end(self, outputs):

        preds, batch = outputs

        # get one sample from batch
        pred = preds[0].squeeze()*255.0
        image = batch['image'][0]
        label = batch["alpha"][0].squeeze()*255.0
        trimap = batch["trimap"][0].squeeze()*128

        dataset_name = batch["dataset_name"][0]
        image_name = batch["image_name"][0].split('.')[0]

        self.__compute_and_log_mse_sad_of_one_sample(
            pred, label, trimap, prefix="val")

        self.__log_image(
            image, pred, label, trimap, dataset_name, image_name, prefix='val')

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

    def __log_image(self, image, pred, label, trimap, dataset_name, image_name, prefix='val'):
        ########### log source_image, masked_reference_image, output and gt ###########
        # process image, masked_reference_image, pred, label
        image = self.__revert_normalize(image)
        pred = torch.stack((pred/255.0,)*3, axis=-1)
        label = torch.stack((label/255.0,)*3, axis=-1)
        trimap = torch.stack((trimap/255.0,)*3, axis=-1)

        # concat pred, masked_reference_image, label, source_image
        image_for_log = torch.stack(
            (image, trimap, label, pred), axis=0)

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
        params = params + self.decoder.get_trainable_params() + \
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


class ModifyModelSave(pl.Callback):
    # unused
    # TODO: state_dict contains no clip and diffusion model, but why?
    # TODO: move to callbacks.py

    def delete_frozen_params(self, ckpt):

        # delete params with requires_grad=False
        for k, v in ckpt['state_dict'].items():
            if not v.requires_grad:
                del ckpt['state_dict'][k]
        return ckpt

    def on_pretrain_routine_end(self, trainer, pl_module):
        model_checkpoint = trainer.checkpoint_callback

        def _save_model_del_frozen_params(self, trainer: "pl.Trainer", filepath: str) -> None:
            model_checkpoint._save_model(trainer, filepath)
            # load params, delete frozen params, save params
            ckpt = torch.load(filepath)

            # delete frozen params
            ckpt = self.delete_frozen_params(ckpt)
            torch.save(ckpt, filepath)

        model_checkpoint._save_model = _save_model_del_frozen_params

    # def _save_model(self, trainer: "pl.Trainer", filepath: str) -> None:
    #     # in debugging, track when we save checkpoints
    #     trainer.dev_debugger.track_checkpointing_history(filepath)

    #     # make paths
    #     if trainer.should_rank_save_checkpoint:
    #         self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

    #     # delegate the saving to the trainer
    #     trainer.save_checkpoint(filepath, self.save_weights_only)
