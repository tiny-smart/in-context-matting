import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pytorch_lightning import Trainer, seed_everything
import torch
from icm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import datetime
# import tensorboard

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="diffusion_matte-train_adapter_params_True",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
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
    cfg = OmegaConf.load(args.config)
    
    # set seed
    seed_everything(args.seed)

    """=== Init data ==="""
    cfg_data = cfg.get('data')

    data = instantiate_from_config(cfg_data)
    
    """=== Init model ==="""
    cfg_model = cfg.get('model')

    model = instantiate_from_config(cfg_model)

    """=== Init trainer ==="""
    cfg_trainer = cfg.get('trainer')
    # omegaconf to dict
    cfg_trainer = OmegaConf.to_container(cfg_trainer)
    
    if args.debug:
        cfg_trainer['limit_train_batches'] = 2
        # cfg_trainer['log_every_n_steps'] = 1
        cfg_trainer['limit_val_batches'] = 3
        # cfg_trainer['overfit_batches'] = 10
    
    # init logger
    cfg_logger = cfg_trainer.pop('cfg_logger')
    
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'-'+args.experiment_name
    cfg_logger['params']['save_dir'] = args.logdir
    cfg_logger['params']['name'] = name
    cfg_trainer['logger'] = instantiate_from_config(cfg_logger)
    
    # init callbacks
    cfg_callbacks = cfg_trainer.pop('cfg_callbacks')
    callbacks = []
    for callback_name in cfg_callbacks:
        if callback_name == 'modelcheckpoint':
            cfg_callbacks[callback_name]['params']['dirpath'] = os.path.join(args.logdir, name,'checkpoints')
        callbacks.append(instantiate_from_config(cfg_callbacks[callback_name]))
    cfg_trainer['callbacks'] = callbacks

    # init trainer
    trainer_opt = argparse.Namespace(**cfg_trainer)
    trainer = Trainer.from_argparse_args(trainer_opt)

    # save configs to log
    trainer.logger.log_hyperparams(cfg)
    
    """=== Start training ==="""

    trainer.fit(model, data)
