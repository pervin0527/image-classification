import os
import cv2
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.plot_util import visualize_class_distribution

def train_valid_split(data_path, test_size, random_state):
    df = pd.read_csv(f"{data_path}/train.csv")
    meta_df = pd.read_csv(f"{data_path}/meta.csv")

    train_data, valid_data = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['target'])
    print(train_data.shape, valid_data.shape)

    os.makedirs("./imgs", exist_ok=True)
    visualize_class_distribution(train_data, meta_df, save_plot=True, plot_path="./imgs/train_dist.png")
    visualize_class_distribution(valid_data, meta_df, save_plot=True, plot_path="./imgs/valid_dist.png")

    train_data.drop(columns=['class_name'], axis=1, inplace=True)
    valid_data.drop(columns=['class_name'], axis=1, inplace=True)

    train_data.to_csv(f'{data_path}/train-data.csv', index=False)
    valid_data.to_csv(f'{data_path}/valid-data.csv', index=False)


def compute_mean_std(csv_path, image_path, img_size=256, save_path='mean_std.pkl', sample_size=None):
    """
    CSV 파일에서 이미지 목록을 읽고, 데이터셋의 평균(mean)과 표준 편차(std)를 계산하거나 불러오는 함수.
    
    Args:
        csv_path (str): 이미지 파일 목록이 포함된 CSV 파일 경로.
        image_path (str): 이미지 파일이 저장된 디렉토리 경로.
        save_path (str): 계산된 값을 저장하거나 불러올 파일 경로 (기본값: 'mean_std.pkl').
        sample_size (int, optional): 샘플링할 이미지 개수 (기본값: None, 전체 이미지 사용).

    Returns:
        tuple: (mean, std)
    """
    if os.path.exists(save_path):
        # 파일이 존재하면 값을 불러옴
        with open(save_path, 'rb') as f:
            mean, std = pickle.load(f)
        print("Loaded mean and std from file.")
    else:
        # 파일이 존재하지 않으면 값을 계산하고 저장
        df = pd.read_csv(csv_path)
        if 'ID' not in df.columns:
            raise ValueError("CSV 파일에 'ID' 컬럼이 없습니다.")
        
        image_files = df['ID'].tolist()
        
        if sample_size:
            np.random.seed(42)
            image_files = np.random.choice(image_files, sample_size, replace=False)

        num_channels = 3
        channel_sum = np.zeros(num_channels)
        channel_sum_squared = np.zeros(num_channels)
        num_pixels = 0

        for image_file in tqdm(image_files, desc="Calculating mean and std"):
            img_path = os.path.join(image_path, image_file)
            img = cv2.imread(img_path)
            if img is None:
                continue  # 이미지를 읽지 못한 경우 건너뜀
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_size:
                img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            img = img.reshape(-1, num_channels)
            
            channel_sum += img.sum(axis=0)
            channel_sum_squared += (img ** 2).sum(axis=0)
            num_pixels += img.shape[0]

        mean = channel_sum / num_pixels
        std = np.sqrt(channel_sum_squared / num_pixels - mean ** 2)

        # 계산된 값을 파일에 저장
        with open(save_path, 'wb') as f:
            pickle.dump((mean, std), f)
        print("Saved mean and std to file.")

    return mean.tolist(), std.tolist()

class ClassificationDataset(Dataset):
    def __init__(self, csv_path, meta_path, img_path, transform=None):
        self.img_path = img_path
        self.df = pd.read_csv(csv_path).sample(frac=1).values
        self.transform = transform

        meta_df = pd.read_csv(meta_path)
        self.classes = meta_df['class_name'].tolist()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name, target = self.df[idx]
        image = cv2.imread(f"{self.img_path}/{file_name}")

        if image is not None:
            image = np.asarray(image)
        else:
            raise ValueError(f"Error loading image {self.img_path}/{file_name}")

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 필요시 사용

        if self.transform:
            image = self.transform(image=image)['image']

        return image, target
