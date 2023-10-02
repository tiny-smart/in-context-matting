if __name__ == '__main__':
    import datetime
    import argparse
    from omegaconf import OmegaConf

    import os
    # set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    from icm.util import instantiate_from_config
    import torch
    from pytorch_lightning import Trainer, seed_everything
    # import tensorboard


    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--experiment_name",
            type=str,
            # "diffusion_matte-train_adapter_params_True-bs_2",
            # "in_context_matting-0.1",
            default="in_context_matting",
        )
        parser.add_argument(
            "--debug",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--resume",
            type=str,
            # default='logs/2023-09-12_19-21-22-in_context_matting-1.0-2waytransformer_norm_ff_4lr/checkpoints/10-0.01410-0.03703.ckpt',
            # default="logs/2023-09-27_11-43-30-in_context_matting-openimages-l1loss-1waytrans-kvembed-deft/checkpoints/01-0.07059-0.22562.ckpt",
            default="logs/2023-10-01_12-45-16-in_context_matting/checkpoints/07-0.05928.ckpt",
        )
        parser.add_argument(
            "--fine_tune",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--config",
            type=str,
            # default="config/train_diffusion_matting.yaml"
            # default="config/train_in_context_matting_transformer.yaml",
            default="config/train_in_context_matting_correspondence.yaml",
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

    import multiprocessing
    multiprocessing.set_start_method('spawn')
    # ... **The all rest code**
    
    args = parse_args()
    if args.resume:
        path = args.resume.split('checkpoints')[0]
        # get the folder of last version folder
        all_folder = os.listdir(path)
        all_folder = [os.path.join(path, folder) for folder in all_folder if 'version' in folder]
        all_folder.sort()
        last_version_folder = all_folder[-1]
        # get the hparams.yaml path
        hparams_path = os.path.join(last_version_folder, 'hparams.yaml')
        cfg = OmegaConf.load(hparams_path)
    else:
        cfg = OmegaConf.load(args.config)

    if args.fine_tune:
        cfg_ft = OmegaConf.load(args.config)
        # merge cfg and cfg_ft, cfg_ft will overwrite cfg
        cfg = OmegaConf.merge(cfg, cfg_ft)
        
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
        # cfg_trainer['limit_val_batches'] = 3
        # cfg_trainer['overfit_batches'] = 2

    # init logger
    cfg_logger = cfg_trainer.pop('cfg_logger')

    if args.resume:
        name = args.resume.split('/')[-3]
    else:
        name = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")+'-'+args.experiment_name
    cfg_logger['params']['save_dir'] = args.logdir
    cfg_logger['params']['name'] = name
    cfg_trainer['logger'] = instantiate_from_config(cfg_logger)

    # plugin
    cfg_plugin = cfg_trainer.pop('plugins')
    cfg_trainer['plugins'] = instantiate_from_config(cfg_plugin)
    
    # init callbacks
    cfg_callbacks = cfg_trainer.pop('cfg_callbacks')
    callbacks = []
    for callback_name in cfg_callbacks:
        if callback_name == 'modelcheckpoint':
            cfg_callbacks[callback_name]['params']['dirpath'] = os.path.join(
                args.logdir, name, 'checkpoints')
        callbacks.append(instantiate_from_config(cfg_callbacks[callback_name]))
    cfg_trainer['callbacks'] = callbacks

    if args.resume:
        cfg_trainer['resume_from_checkpoint'] = args.resume
    
    # init trainer
    trainer_opt = argparse.Namespace(**cfg_trainer)
    trainer = Trainer.from_argparse_args(trainer_opt)

    # save configs to log
    trainer.logger.log_hyperparams(cfg)

    """=== Start training ==="""

    trainer.fit(model, data)
