import os
import timm
import torch

from tqdm import tqdm
from datetime import datetime

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

from utils.common_util import parse_args
from utils.config_util import load_config, save_config
from utils.train_util import make_save_dir, save_batch_images

from data.dataset import ClassificationDataset
from data.augmentation import train_transform, eval_transform

def train(model, dataloader, optimizer, loss_func, device, writer, epoch):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_func(preds, labels)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(labels.detach().cpu().numpy())

        train_loss /= len(dataloader)
        train_acc = accuracy_score(targets_list, preds_list)
        train_f1 = f1_score(targets_list, preds_list, average='macro')

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('F1_Score/train', train_f1, epoch)

    result = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return result


def main(cfg):
    save_dir = make_save_dir(cfg['save_path'])
    writer = SummaryWriter(log_dir=f"{save_dir}/logs")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ClassificationDataset(csv_path=f"{cfg['data_path']}/train.csv", 
                                          meta_path=f"{cfg['data_path']}/meta.csv",
                                          img_path=f"{cfg['data_path']}/train", 
                                          transform=train_transform(cfg['img_size']))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    classes = train_dataset.classes

    if cfg['save_batch_imgs']:
        for batch_idx, data in enumerate(train_dataloader):
            # print(f"Processing batch {batch_idx+1}")
            save_batch_images(data, output_dir="./datasets/batch_images")

            break

    model = timm.create_model(cfg['model_name'], pretrained=True, num_classes=len(classes)).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    for epoch in range(1, cfg['epochs']):
        print(f"Epoch [{epoch} | {cfg['epochs']}]")
        
        train_result = train(model, train_dataloader, optimizer, loss_func, device, writer, epoch)
        print(f"Train Loss : {train_result['train_loss']:.4f}, Train Acc : {train_result['train_acc']:.4f}, Train F1 : {train_result['train_f1']:.4f}")

        print()

    writer.close()

    
if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)
    main(cfg)