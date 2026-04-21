import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COL = "target"
DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]


class CriteoDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        # 数值特征：float32
        self.dense_features = torch.tensor(
            self.df[DENSE_FEATURES].values,
            dtype=torch.float32
        )

        # 类别特征：long，后面 embedding 要用
        self.sparse_features = torch.tensor(
            self.df[SPARSE_FEATURES].values,
            dtype=torch.long
        )

        # 标签：float32，后面做二分类损失
        self.labels = torch.tensor(
            self.df[TARGET_COL].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "dense_x": self.dense_features[idx],
            "sparse_x": self.sparse_features[idx],
            "label": self.labels[idx]
        }