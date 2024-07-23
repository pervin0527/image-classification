import os
from datetime import datetime
from torchvision.utils import save_image

def make_save_dir(save_path):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(save_path, timestamp)
    os.makedirs(f"{save_dir}/weights", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    return save_dir


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 반대로 정규화를 되돌림
    return tensor


def save_batch_images(data, output_dir="output_images"):
    images, labels = data

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for idx, (image, label) in enumerate(zip(images, labels)):
        img = denormalize(image, mean, std)        
        img = img.clamp(0, 1)

        image_filename = os.path.join(output_dir, f"image_{idx}_label_{label.item()}.png")
        save_image(img, image_filename)
        # print(f"Saved {image_filename}")