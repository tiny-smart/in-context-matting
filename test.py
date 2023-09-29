import datetime
import argparse
from omegaconf import OmegaConf
from icm.util import instantiate_from_config
import torch
from pytorch_lightning import Trainer, seed_everything
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2,"

# import tensorboard


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
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
        # default='logs/2023-09-12_19-21-22-in_context_matting-1.0-2waytransformer_norm_ff_4lr/checkpoints/10-0.01410-0.03703.ckpt',
        default="logs/2023-09-27_11-43-30-in_context_matting-openimages-l1loss-1waytrans-kvembed-deft/checkpoints/13-0.06500-0.21785.ckpt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/test.yaml",
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
        all_folder = [os.path.join(path, folder)
                      for folder in all_folder if 'version' in folder]
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

    # model = instantiate_from_config(cfg_model)
    model = load_model_from_config(cfg_model, args.checkpoint, verbose=True)

    """=== Start validation ==="""
    model.on_train_start()
    model.eval()
    model.cuda()
    # model.train()
    loss_list = []
    for batch in tqdm(data._val_dataloader()):
        # move tensor in batch to cuda
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        output, loss = model.test_step(batch, None)
        loss_list.append(loss.item())
        print('Validation loss: ', sum(loss_list)/len(loss_list))
    print('Validation loss: ', sum(loss_list)/len(loss_list))
    print('Finish validation')