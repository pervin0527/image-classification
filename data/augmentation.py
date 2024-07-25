import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transform(img_size, mean, std):
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return transform

def eval_transform(img_size, mean, std):
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return transform