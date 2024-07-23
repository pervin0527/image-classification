import cv2
import pandas as pd
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        self.img_path = img_path
        self.df = pd.read_csv(csv_path).sample(frac=1).values
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name, target = self.df[idx]
        image = cv2.imread(f"{self.img_path}/{file_name}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, target
    

if __name__ == "__main__":
    data_path = "../datasets"
    dataset = ClassificationDataset(f"{data_path}/train.csv", 
                                    f"{data_path}/train")
    
    sample_img, sample_label = dataset[0]
    cv2.imwrite("./sample.jpg", sample_img)
    print(sample_label)
