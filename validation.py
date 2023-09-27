import datetime
import argparse
from omegaconf import OmegaConf
from icm.util import instantiate_from_config
import torch
from pytorch_lightning import Trainer, seed_everything
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,"

# import tensorboard


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        # default='logs/2023-09-12_19-21-22-in_context_matting-1.0-2waytransformer_norm_ff_4lr/checkpoints/10-0.01410-0.03703.ckpt',
        default="logs/2023-09-21_23-48-08-in_context_matting-openimages-l1loss-1waytrans-4head-2data/checkpoints/54-0.07293-0.21500.ckpt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/validation.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.checkpoint:
        path = args.checkpoint.split('checkpoints')[0]
        # get the folder of last version folder
        all_folder = os.listdir(path)
        all_folder = [os.path.join(path, folder) for folder in all_folder if 'version' in folder]
        all_folder.sort()
        last_version_folder = all_folder[-1]
        # get the hparams.yaml path
        hparams_path = os.path.join(last_version_folder, 'hparams.yaml')
        cfg = OmegaConf.load(hparams_path)
    else:
        raise ValueError('Please input the checkpoint path')

    # set seed
    seed_everything(args.seed)

    """=== Init data ==="""
    
    cfg_data = OmegaConf.load(args.config).get('data')

    data = instantiate_from_config(cfg_data)
    data.setup()
    
    """=== Init model ==="""
    cfg_model = cfg.get('model')

    model = instantiate_from_config(cfg_model)

    """=== Init trainer ==="""
    cfg_trainer = cfg.get('trainer')
    # modify gpus to 1
    cfg_trainer['gpus'] = 1
    # omegaconf to dict
    cfg_trainer = OmegaConf.to_container(cfg_trainer)

    # if args.debug:
    #     cfg_trainer['limit_train_batches'] = 2
    #     # cfg_trainer['log_every_n_steps'] = 1
    #     # cfg_trainer['limit_val_batches'] = 3
    #     # cfg_trainer['overfit_batches'] = 10

    # # init logger
    # cfg_logger = cfg_trainer.pop('cfg_logger')

    # if args.checkpoint:
    #     name = args.checkpoint.split('/')[-3]
    # else:
    #     name = datetime.datetime.now().strftime(
    #         "%Y-%m-%d_%H-%M-%S")+'-'+args.experiment_name
    # cfg_logger['params']['save_dir'] = args.logdir
    # cfg_logger['params']['name'] = name
    # cfg_trainer['logger'] = instantiate_from_config(cfg_logger)

    # plugin
    cfg_plugin = cfg_trainer.pop('plugins')
    cfg_trainer['plugins'] = instantiate_from_config(cfg_plugin)
    
    # # init callbacks
    # cfg_callbacks = cfg_trainer.pop('cfg_callbacks')
    # callbacks = []
    # for callback_name in cfg_callbacks:
    #     if callback_name == 'modelcheckpoint':
    #         cfg_callbacks[callback_name]['params']['dirpath'] = os.path.join(
    #             args.logdir, name, 'checkpoints')
    #     callbacks.append(instantiate_from_config(cfg_callbacks[callback_name]))
    # cfg_trainer['callbacks'] = callbacks

    if args.checkpoint:
        cfg_trainer['resume_from_checkpoint'] = args.checkpoint
    
    # init trainer
    trainer_opt = argparse.Namespace(**cfg_trainer)
    trainer = Trainer.from_argparse_args(trainer_opt)

    # # save configs to log
    # trainer.logger.log_hyperparams(cfg)

    """=== Start validation ==="""
    model.on_train_start()
    trainer.validate(model=model, datamodule=data)
    print('Finish validation')
