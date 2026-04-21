from torch.utils.data import DataLoader
from dataset import CriteoDataset

TRAIN_PATH = "data/processed/train_processed.csv"


def main():
    dataset = CriteoDataset(TRAIN_PATH)
    print(f"数据集样本数: {len(dataset)}")

    sample = dataset[0]
    print("单条样本的字段：", sample.keys())
    print("dense_x shape:", sample["dense_x"].shape)
    print("sparse_x shape:", sample["sparse_x"].shape)
    print("label:", sample["label"])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    print("\n一个 batch 的形状：")
    print("dense_x shape:", batch["dense_x"].shape)
    print("sparse_x shape:", batch["sparse_x"].shape)
    print("label shape:", batch["label"].shape)


if __name__ == "__main__":
    main()