import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
def train_transform(img_size, mean, std):
    transform = A.Compose([
        A.OneOf([
            A.Resize(img_size, img_size, p=0.5),
            A.RandomResizedCrop(img_size, img_size, p=0.5)
        ], p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25), p=1),

        A.Rotate(limit=(-90, 90), border_mode=0),
        
        A.OneOf([
            A.ElasticTransform(border_mode=0, p=0.5),
            A.OpticalDistortion(border_mode=0, p=0.5)
        ], p=0.5),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.OneOf([
            A.Blur(blur_limit=(3, 5), p=0.33),
            A.GaussianBlur(blur_limit=(3, 5), p=0.33),
            A.MotionBlur(blur_limit=(3, 5), p=0.33)
        ], p=0.5),

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