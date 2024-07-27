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
from utils.train_util import set_seed, make_save_dir, save_batch_images

from data.dataset import ClassificationDataset, compute_mean_std, train_valid_split
from data.augmentation import train_transform, eval_transform


def valid(model, dataloader, loss_func, device, writer, epoch):
    model.eval()
    valid_loss = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Valid", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = loss_func(preds, labels)
            valid_loss += loss.item() * images.size(0)

            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(labels.argmax(dim=1).detach().cpu().numpy())

    valid_loss /= len(dataloader)
    valid_acc = accuracy_score(targets_list, preds_list)
    valid_f1 = f1_score(targets_list, preds_list, average='macro')

    writer.add_scalars('valid', {'Loss': valid_loss}, epoch)
    writer.add_scalars('valid', {'Accuracy': valid_acc}, epoch)
    writer.add_scalars('valid', {'F1_Score': valid_f1}, epoch)

    result = {
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
    }

    return result


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

        train_loss += loss.item() * images.size(0)

        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(labels.argmax(dim=1).detach().cpu().numpy())

    train_loss /= len(dataloader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    writer.add_scalars('train', {'Loss': train_loss}, epoch)
    writer.add_scalars('train', {'Accuracy': train_acc}, epoch)
    writer.add_scalars('train', {'F1_Score': train_f1}, epoch)

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

    if not os.path.exists(f"{cfg['data_path']}/train-data.csv") and not os.path.exists(f"{cfg['data_path']}/valid-data.csv"):
        train_valid_split(f"{cfg['data_path']}", cfg['valid_ratio'], cfg['seed'])

    mean, std = compute_mean_std(csv_path=f"{cfg['data_path']}/train-data.csv",
                                 image_path=f"{cfg['data_path']}/train",
                                 img_size=cfg['img_size'],
                                 save_path=f"{cfg['data_path']}/mean_std.pkl")
    
    train_dataset = ClassificationDataset(cfg, is_train=True, transform=train_transform(cfg['img_size'], mean, std))
    valid_dataset = ClassificationDataset(cfg, is_train=False, transform=eval_transform(cfg['img_size'], mean, std))
    print(len(train_dataset), len(valid_dataset))    

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    classes = train_dataset.classes

    if cfg['save_batch_imgs']:
        for batch_idx, data in enumerate(train_dataloader):
            images, labels = data
            print(images.shape, labels.shape)
            save_batch_images(data, output_dir="./datasets/batch_images")
            break

    model = timm.create_model(cfg['model_name'], pretrained=True, num_classes=len(classes), strict=False).to(device)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['reduce_factor'], patience=cfg['reduce_patience'])

    save_config(cfg, save_dir)
    best_valid_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = cfg.get('early_stop_patience', 10)
    for epoch in range(1, cfg['epochs'] + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch} | {cfg['epochs']}], LR : {current_lr}")
        
        train_result = train(model, train_dataloader, optimizer, loss_func, device, writer, epoch)
        print(f"Train Loss : {train_result['train_loss']:.4f}, Train Acc : {train_result['train_acc']:.4f}, Train F1 : {train_result['train_f1']:.4f}")

        valid_result = valid(model, valid_dataloader, loss_func, device, writer, epoch)
        print(f"Valid Loss : {valid_result['valid_loss']:.4f}, Valid Acc : {valid_result['valid_acc']:.4f}, Valid F1 : {valid_result['valid_f1']:.4f}")

        scheduler.step(valid_result['valid_loss'])
        if valid_result['valid_loss'] < best_valid_loss:
            print(f"Valid Loss Updated | prev : {best_valid_loss:.4f} --> cur : {valid_result['valid_loss']}")
            best_valid_loss = valid_result['valid_loss']
            torch.save(model.state_dict(), os.path.join(save_dir, 'weights', 'best.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Valid Loss Not Updated | early_stop_counter : {early_stopping_counter}")

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping")
            break

        print()

    torch.save(model.state_dict(), os.path.join(save_dir, 'weights', 'last.pth'))
    writer.close()

    
if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    set_seed(cfg['seed'])
    main(cfg)
