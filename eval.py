import datetime
import argparse
from omegaconf import OmegaConf
from icm.util import instantiate_from_config
import torch
from pytorch_lightning import Trainer, seed_everything
import os
from tqdm import tqdm

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
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
    # if args.checkpoint:
    #     path = args.checkpoint.split('checkpoints')[0]
    #     # get the folder of last version folder
    #     all_folder = os.listdir(path)
    #     all_folder = [os.path.join(path, folder)
    #                   for folder in all_folder if 'version' in folder]
    #     all_folder.sort()
    #     last_version_folder = all_folder[-1]
    #     # get the hparams.yaml path
    #     hparams_path = os.path.join(last_version_folder, 'hparams.yaml')
    #     cfg = OmegaConf.load(hparams_path)
    # else:
    #     raise ValueError('Please input the checkpoint path')

    # set seed
    seed_everything(args.seed)

    cfg = OmegaConf.load(args.config)
    
    """=== Init data ==="""
    
    cfg_data = cfg.get('data')

    data = instantiate_from_config(cfg_data)
    data.setup()

    """=== Init model ==="""
    cfg_model = cfg.get('model')

    # model = instantiate_from_config(cfg_model)
    model = load_model_from_config(cfg_model, args.checkpoint, verbose=True)

    """=== Start validation ==="""
    model.on_train_start()
    model.eval()
    model.cuda()
    # model.train()
    # loss_list = []
    # for batch in tqdm(data._val_dataloader()):
    #     # move tensor in batch to cuda
    #     for key in batch:
    #         if isinstance(batch[key], torch.Tensor):
    #             batch[key] = batch[key].cuda()
    #     output, loss = model.test_step(batch, None)
    #     loss_list.append(loss.item())
    #     print('Validation loss: ', sum(loss_list)/len(loss_list))
    # print('Validation loss: ', sum(loss_list)/len(loss_list))
    # print('Finish validation')


    # init trainer for validation
    cfg_trainer = cfg.get('trainer')
    # set gpu = 1
    cfg_trainer.gpus = 1


    # omegaconf to dict
    cfg_trainer = OmegaConf.to_container(cfg_trainer)
    cfg_trainer.pop('cfg_callbacks') if 'cfg_callbacks' in cfg_trainer else None
    # init logger
    cfg_logger = cfg_trainer.pop('cfg_logger') if 'cfg_logger' in cfg_trainer else None
    cfg_logger['params']['save_dir'] = 'logs/'
    cfg_logger['params']['name'] = 'eval'
    cfg_trainer['logger'] = instantiate_from_config(cfg_logger)

    # plugin
    cfg_plugin = cfg_trainer.pop('plugins') if 'plugins' in cfg_trainer else None

    # init trainer
    trainer_opt = argparse.Namespace(**cfg_trainer)
    trainer = Trainer.from_argparse_args(trainer_opt)
    # init logger
    model.val_save_path = args.save_path
    trainer.validate(model, data.val_dataloader())
    