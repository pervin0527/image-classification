import os
import timm
import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.train_util import set_seed
from utils.common_util import parse_args
from utils.config_util import load_config
from data.augmentation import eval_transform
from data.dataset import ClassificationDataset, compute_mean_std


def load_model(model_path, model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def inference(model, dataloader, device, threshold=0.5):
    preds_list = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds)
            preds_list.extend(preds.cpu().numpy())
    return preds_list


def convert_predictions_to_string(preds, class_names, threshold=0.5):
    pred_strings = []
    for pred in preds:
        pred_classes = [class_names[i] for i, prob in enumerate(pred) if prob >= threshold]
        pred_strings.append(" ".join(pred_classes))
    return pred_strings


def save_predictions(pred_strings, output_path):
    df = pd.DataFrame(pred_strings, columns=['prediction'])
    df.to_csv(output_path, index_label='id')


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std = compute_mean_std(cfg, save_path=f"{cfg['data_path']}/mean_std.pkl")
    test_dataset = ClassificationDataset(cfg, ds_type='test', transform=eval_transform(cfg['img_size'], mean, std))
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    model = load_model(cfg['weight_path'], cfg['model_name'], len(test_dataset.classes), device)
    preds = inference(model, test_dataloader, device, cfg.get('threshold', 0.5))
    pred_strings = convert_predictions_to_string(preds, test_dataset.classes, cfg.get('threshold', 0.5))
    save_predictions(pred_strings, cfg['output_path'])


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    set_seed(cfg['seed'])
    main(cfg)
