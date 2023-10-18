import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(annotations_file, header=None)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)  # 这玩意儿应该对计算dataloader的长度有用

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])  # 拿到图像的路径
        image = read_image(image_path)  # 拿到图像（直接就是tensor，不需要转换格式）
        label = self.image_labels.iloc[idx, 1]  # 拿到标签
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    training_data = CustomDataset('images_labels.csv', './datasets/images')
    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))  # 分别是tensor和tuple
    print(f"Feature shape and slice value: {train_features.size(), train_features[0][0]}")
    print(f"Feature other info: {train_features.min(), train_features.max(), train_features.dtype}")
    print(f"Labels value: {train_labels, type(train_labels[0])}")
    # 关于dataloader吐出的数据：
    # 1. image是tensor，没有规一化=>范围是[0, 255]，类型是uint8
    # 2. label是tuple，里面的元素（标签值）类型一般是str（数字也行）
