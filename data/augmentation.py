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


def rand_bbox(size, lam):
    W = size[1]
    H = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(image1, image2, label1, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image1.shape, lam)
    
    image1[bbx1:bbx2, bby1:bby2, :] = image2[bbx1:bbx2, bby1:bby2, :]
    label = lam * label1 + (1 - lam) * label2
    
    return image1, label


def mixup(image1, image2, label1, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    
    mixup_image = lam * image1 + (1 - lam) * image2
    mixup_image = mixup_image.astype(np.uint8)
    
    mixup_label = lam * label1 + (1 - lam) * label2
    
    return mixup_image, mixup_label


def cutout(image, mask_size, p=0.5):
    if np.random.rand() > p:
        return image
    
    h, w = image.shape[:2]
    mask_size_half = mask_size // 2

    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    x1 = np.clip(cx - mask_size_half, 0, w)
    y1 = np.clip(cy - mask_size_half, 0, h)
    x2 = np.clip(cx + mask_size_half, 0, w)
    y2 = np.clip(cy + mask_size_half, 0, h)

    image[y1:y2, x1:x2, :] = 0
    
    return image