from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from icm.models.criterion.matting_criterion import MattingCriterion
from icm.models.criterion.matting_criterion_eval import compute_mse_loss, compute_sad_loss, compute_connectivity_error, compute_gradient_loss
from icm.util import instantiate_from_config, instantiate_feature_extractor
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class DiffusionMatting(pl.LightningModule):
    def __init__(
        self,
        cfg_feature_extractor,
        cfg_decoder,
        feature_index,
        learning_rate,
        guidance_type,
        use_scheduler,
        scheduler_config,
    ):
        super().__init__()

        # init model, move to configure_shared_model, move back to init
        self.feature_extractor = instantiate_feature_extractor(
            cfg_feature_extractor)

        self.diffusion_decoder = instantiate_from_config(cfg_decoder)
        # self.cfg_feature_extractor = cfg_feature_extractor
        # self.cfg_decoder = cfg_decoder

        self.feature_index = feature_index
        self.guidance_type = guidance_type
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.scheduler_config = scheduler_config

        self.criterion = MattingCriterion(
            losses=[
                "unknown_l1_loss",
                "known_l1_loss",
                "loss_pha_laplacian",
                "loss_gradient_penalty",
            ]
        )

    # def configure_sharded_model(self):
    #     self.feature_extractor = instantiate_feature_extractor(
    #         self.cfg_feature_extractor
    #     )
    #     self.diffusion_decoder = instantiate_from_config(self.cfg_decoder)

    def on_train_start(self):
        # set layers to get features
        self.feature_extractor.reset_dim_stride()

    def on_train_epoch_start(self, unused=None):
        self.log("epoch", self.current_epoch, on_step=False,
                    on_epoch=True, prog_bar=False)
        
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, img, images_guidance):
        x = self.feature_extractor({'img': img})[self.feature_index].detach()
        x = self.diffusion_decoder(x, images_guidance)

        return x

    def get_guidance_map(self, trimaps, labels):
        if self.guidance_type == "coarse_map":
            # make coarse mask, 0 for bg, 1 for fg and unknown
            coarse_mask = torch.zeros_like(trimaps)
            coarse_mask[trimaps == 1] = 1
            coarse_mask[labels > 0.5] = 1
            return coarse_mask

        elif self.guidance_type == "trimap":
            return trimaps

        elif self.guidance_type == "null":
            return torch.zeros_like(trimaps)

        else:
            raise NotImplementedError

    def shared_step(self, batch, batch_idx):
        images, labels, trimaps = batch["image"], batch["alpha"], batch["trimap"]

        guidance_map = self.get_guidance_map(trimaps, labels)
        images_guidance = torch.cat((images, guidance_map), dim=1)

        output = self(images, images_guidance)
        return output

    def training_step(self, batch, batch_idx):
        labels, trimaps = batch["alpha"], batch["trimap"]

        output = self.shared_step(batch, batch_idx)

        sample_map = torch.zeros_like(trimaps)
        sample_map[trimaps == 1] = 1
        losses = self.criterion(sample_map, {"phas": output}, {"phas": labels})

        # log training loss
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=False)

        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]
                 ["lr"], on_step=True, on_epoch=False, prog_bar=True)

        # init loss tensor
        loss = torch.zeros(1).type_as(labels)
        
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        for key in losses:
            loss += losses[key]

        return loss

    def validation_step(self, batch, batch_idx):
        # batch size = 1
        assert batch["image"].shape[0] == 1
        
        labels, trimaps = batch["alpha"], batch["trimap"]

        output = self.shared_step(batch, batch_idx)

        label = labels.squeeze().cpu().numpy()*255.0

        trimap = trimaps.squeeze().cpu().numpy()*128

        pred = output.squeeze().cpu().numpy()*255.0

        # compute loss

        # compute loss for unknown pixels
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(
            pred, label, trimap)[0]
        conn_loss_unknown_ = compute_connectivity_error(
            pred, label, trimap, 0.1)
        grad_loss_unknown_ = compute_gradient_loss(
            pred, label, trimap)

        # compute loss for all pixels
        trimap = np.ones_like(label)*128

        mse_loss_all_ = compute_mse_loss(pred, label, trimap)
        sad_loss_all_ = compute_sad_loss(
            pred, label, trimap)[0]
        conn_loss_all_ = compute_connectivity_error(
            pred, label, trimap, 0.1)
        grad_loss_all_ = compute_gradient_loss(
            pred, label, trimap)

        # log validation metrics
        metrics_unknown = {'mse_unknown': mse_loss_unknown_,
                           'sad_unknown': sad_loss_unknown_,
                           'conn_unknown': conn_loss_unknown_,
                           'grad_unknown': grad_loss_unknown_}

        metrics_all = {'mse_all': mse_loss_all_,
                       'sad_all': sad_loss_all_,
                       'conn_all': conn_loss_all_,
                       'grad_all': grad_loss_all_}

        self.log_dict(metrics_unknown, on_step=False,
                      on_epoch=True, prog_bar=False)
        self.log_dict(metrics_all, on_step=False,
                      on_epoch=True, prog_bar=False)
        # self.logger.experiment.add_scalars('metrics_unknown', metrics_unknown)
        # self.logger.experiment.add_scalars('metrics_all', metrics_all)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.diffusion_decoder.parameters()

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

# TODO: move to callbacks.py


class ModifyModelSave(pl.Callback):
    # TODO: state_dict contains no clip and diffusion model, but why?
    
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
