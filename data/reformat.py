import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    data_path = "../datasets/PlantVillage"
    classes = sorted(os.listdir(data_path))

    class_dict = {}
    for idx, label in enumerate(classes):
        class_dict[idx] = label

    meta_data = [{'target': idx, 'class_name': label} for idx, label in class_dict.items()]
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv("../datasets/meta.csv", index=False)

    total_data_list = []
    for class_idx, label in enumerate(classes):
        files = os.listdir(f"{data_path}/{label}")
        for file in files:
            total_data_list.append({'ID' : file, 'target' : class_idx})
            
    train_df = pd.DataFrame(total_data_list)
    train_df.to_csv("../datasets/train.csv", index=False)

if __name__ == "__main__":
    main()