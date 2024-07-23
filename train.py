import os
import torch
import argparse

from tqdm import tqdm
from datetime import datetime

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.common_util import parse_args
from utils.config_util import load_config, save_config
from utils.train_util import make_save_dir, save_batch_images

from data.dataset import ClassificationDataset
from data.augmentation import train_transform, eval_transform

def main(cfg):
    save_dir = make_save_dir(cfg['save_path'])
    writer = SummaryWriter(log_dir=f"{save_dir}/logs")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ClassificationDataset(f"{cfg['data_path']}/train.csv", f"{cfg['data_path']}/train", train_transform(cfg['img_size']))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    for batch_idx, data in enumerate(train_dataloader):
        print(f"Processing batch {batch_idx+1}")
        save_batch_images(data, output_dir="./datasets/batch_images")

        break
    


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)
    main(cfg)