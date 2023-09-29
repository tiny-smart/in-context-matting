
from icm.models.criterion.matting_criterion_eval import compute_mse_loss, compute_sad_loss, compute_connectivity_error, compute_gradient_loss, compute_mse_loss_torch, compute_sad_loss_torch
from icm.util import instantiate_from_config, instantiate_feature_extractor
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import grad_norm
from torch.nn import functional as F
from torchvision.ops import focal_loss


class InContextMatting(pl.LightningModule):
    '''
    In Context Matting Model
    consists of a feature extractor and a in-context decoder
    train model with lr, scheduler, loss


    feature extractor: # feature_index, 
        list get_trainable_params()  # train_adapter_params
        reset_dim_stride() # reset dim and stride of layers to get features
        get_reference_feature get_source_feature(tensor) :detach()
    in-context decoder: # context_type='maskpooling'
        list get_trainable_params() # freeze_transformer
        forward(source, reference)
                reference = {'feature': feature_of_reference_image,
                     'guidance': guidance_on_reference_image}

        source = {'feature': feature_of_source_image, 'image': source_images}

        x = self.in_context_decoder(source, reference)

                # if self.context_type == 'maskpooling':
        #     feature_of_reference_image = self.context_maskpooling(
        #         feature_of_reference_image, guidance_on_reference_image)
        # elif self.context_type == 'embed':
        #     # resize guidance_on_reference_image to [B, 1, H/d, W/d]
        #     guidance_on_reference_image = F.interpolate(
        #         guidance_on_reference_image, size=feature_of_reference_image.shape[2:], mode='nearest')
        #     # add self.context_embed[0] to pixels where guidance_on_reference_image == 0, add self.context_embed[1] to pixels where guidance_on_reference_image == 1
        #     # embedding = self.context_embed(guidance_on_reference_image.squeeze(1).long()).permute(0, 3, 1, 2)
        #     feature_of_reference_image = feature_of_reference_image + \
        #         self.context_embed(guidance_on_reference_image.squeeze(
        #             1).long()).permute(0, 3, 1, 2)
        #     # flatten feature_of_reference_image
        #     feature_of_reference_image = feature_of_reference_image.reshape(
        #         feature_of_reference_image.shape[0], feature_of_reference_image.shape[1], -1).permute(0, 2, 1)
        # resize guidance_on_reference_image to [B, 1, H/d, W/d]

        guidance_on_reference_image = F.interpolate(
            guidance_on_reference_image, size=feature_of_reference_image.shape[2:], mode='nearest')


    # # unused, move to context_decoder
    # def context_maskpooling(self, feature, mask):
    #     '''
    #     get context feature tokens by maskpooling
    #     feature: [B, C, H/d, W/d]
    #     mask: [B, 1, H, W]  [0,1]
    #     return: [B, token_num, C] token_num = H*W/d^2
    #     '''
    #     mask[mask < 1] = 0
    #     mask = -1 * mask
    #     kernel_size = mask.shape[2] // feature.shape[2]
    #     mask = F.max_pool2d(mask, kernel_size=kernel_size,
    #                         stride=kernel_size, padding=0)
    #     mask = -1*mask

    #     feature = mask*feature
    #     feature = feature.reshape(
    #         feature.shape[0], feature.shape[1], -1).permute(0, 2, 1)

    #     return feature

    # loss_function: # loss_type='vit_matte'
    #         # self.criterion = MattingCriterion(
    #     #     losses=['unknown_l1_loss', 'known_l1_loss',
    #     #             'loss_pha_laplacian', 'loss_gradient_penalty']
    #     # )
    # '''
    #     if self.loss_type == 'vit_matte':
    #     losses = self.criterion(
    #         sample_map, {"phas": output}, {"phas": labels})
    # elif self.loss_type == 'smooth_l1':
    #     losses = F.smooth_l1_loss(output, labels, reduction='none')
    #     losses = {'smooth_l1_loss': losses.mean()}
    # elif self.loss_type == 'cross_entropy':
    #     losses = F.binary_cross_entropy_with_logits(output, labels)
    #     losses = {'cross_entropy': losses.mean()}
    # elif self.loss_type == 'focal_loss':
    #     losses = focal_loss.sigmoid_focal_loss(output, labels)
    #     losses = {'focal_loss': losses.mean()}

    def __init__(
        self,
        cfg_feature_extractor,
        cfg_in_context_decoder,
        cfg_loss_function,
        learning_rate,
        cfg_scheduler=None,
    ):
        super().__init__()

        self.feature_extractor = instantiate_feature_extractor(
            cfg_feature_extractor)
        self.in_context_decoder = instantiate_from_config(
            cfg_in_context_decoder)

        self.loss_function = instantiate_from_config(cfg_loss_function)

        self.learning_rate = learning_rate
        self.cfg_scheduler = cfg_scheduler

    def on_train_start(self):
        # set layers to get features
        self.feature_extractor.reset_dim_stride()

    def on_train_epoch_start(self):
        self.log("epoch", self.current_epoch, on_step=False,
                 on_epoch=True, prog_bar=False, sync_dist=True)

    def forward(self, reference_images, guidance_on_reference_image, source_images):

        feature_of_reference_image = self.feature_extractor.get_reference_feature(
            reference_images)

        feature_of_source_image = self.feature_extractor.get_source_feature(
            source_images)

        reference = {'feature': feature_of_reference_image,
                     'guidance': guidance_on_reference_image}

        source = {'feature': feature_of_source_image, 'image': source_images}

        output = self.in_context_decoder(source, reference)

        return output

    def __shared_step(self, batch):
        reference_images, guidance_on_reference_image, source_images, labels, trimaps = batch[
            "reference_image"], batch["guidance_on_reference_image"], batch["source_image"], batch["alpha"], batch["trimap"]

        outputs = self(reference_images,
                       guidance_on_reference_image, source_images)

        loss_dict = self.loss_function(outputs, labels, trimaps)

        loss = sum(loss_dict.values())
        
        return loss_dict, loss, outputs

    def __log_loss(self, loss_dict, loss, prefix):
        loss_dict = {f"{prefix}/{key}": loss_dict.get(key) for key in loss_dict}
        self.log_dict(loss_dict, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss_dict, loss, _ = self.__shared_step(batch)

        self.__log_loss(loss_dict, loss, "train")
        
        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]
                 ["lr"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict, loss, preds = self.__shared_step(batch)

        self.__log_loss(loss_dict, loss, "val")

        return preds, batch

    def validation_step_end(self, outputs):

        output, batch = outputs
        labels, trimaps, dataset_name, image_name, reference_image, guidance_on_reference_image = batch["alpha"][0].unsqueeze(0), batch[
            "trimap"][0].unsqueeze(0), batch["dataset_name"], batch["image_name"], batch["reference_image"][0].unsqueeze(0), batch["guidance_on_reference_image"][0].unsqueeze(0)

        label = labels.squeeze()*255.0
        trimap = trimaps.squeeze()*128
        pred = output[0].squeeze()*255.0

        # for logging
        guidance_image = reference_image*guidance_on_reference_image
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

    def test_step(self, batch, batch_idx):
        loss_dict, loss, preds = self.__shared_step(batch)

        return preds, loss

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
        params = self.__get_trainable_params()
        opt = torch.optim.Adam(params, lr=self.learning_rate)

        if self.cfg_scheduler is not None:
            scheduler = self.__get_scheduler(opt)
            opt = [opt]
            return opt, scheduler
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
